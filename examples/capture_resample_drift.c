/*
Demonstrates glitch-free capture resampling with device clock drift compensation.

Scenario: the capture device runs at its own native rate (e.g. 44100 Hz) while the
encoder/transport needs a fixed 48000 Hz. Two common mistakes cause trouble here:

    1. Resampling each callback block with a fresh (stateless) converter. The filter
       state is reset on every block which causes discontinuities (clicks) at block
       boundaries.
    2. Assuming the device clock is exactly 44100 Hz. It never is - a capture device
       running 100 ppm fast actually delivers 44104.4 frames per second. If that
       mismatch is not absorbed somewhere it accumulates as ever-growing latency
       (easily several seconds over an hour-long call).

The pattern in this example addresses both:

    - A single ma_data_converter is kept alive for the whole session so the resampler
      carries its filter state across callbacks (no block-boundary artefacts).
    - Converted frames are pushed into a ring buffer which the encoder side drains at
      its own pace. The fill level of that ring buffer is a direct measurement of the
      clock drift: if the device clock is fast the level rises, if it is slow it falls.
    - A slow PI controller nudges the converter's input sample rate by a few Hz
      (clamped to a few hundred ppm) to hold the fill level centered. Rate changes are
      tiny and the resampler keeps its state across ma_data_converter_set_rate(), so
      the correction is inaudible.

The device is opened with sampleRate = 0 so it runs at its native rate and miniaudio
does not insert a hidden conversion stage - this example owns the conversion itself.

The encoder side is simulated by writing 10 ms packets to a WAV file from the main
thread. In a real application you would replace the ma_encoder_write_pcm_frames()
call with your encoder/packetizer.
*/
#include "../miniaudio.c"

#include <stdlib.h>
#include <stdio.h>

#define ENCODER_SAMPLE_RATE     48000
#define ENCODER_CHANNELS        2
#define ENCODER_FORMAT          ma_format_s16
#define PACKET_FRAMES           (ENCODER_SAMPLE_RATE / 100)   /* 10 ms packets. */
#define RB_CAPACITY_FRAMES      (ENCODER_SAMPLE_RATE)         /* 1 second of drift/absorption headroom. */
#define RB_TARGET_FRAMES        (ENCODER_SAMPLE_RATE / 20)    /* Keep the buffer hovering around 50 ms. */

/* Drift controller tuning. Correction is applied to the converter's input rate, in Hz. */
#define DRIFT_KP                0.01f   /* Hz of correction per frame of buffer error. */
#define DRIFT_KI                0.5f    /* Hz of correction per accumulated frame-second of error. */
#define DRIFT_MAX_CORRECTION_HZ 50.0f   /* Clamp to about +/-1100 ppm at 44.1k. */

static ma_data_converter  g_converter;          /* Audio thread only. */
static ma_pcm_rb          g_rb;                 /* Single producer (audio thread), single consumer (main thread). */
static ma_atomic_uint32   g_pendingRateIn;      /* Written by the main thread, applied by the audio thread. */
static ma_uint32          g_baseRateIn;         /* The device's native sample rate, e.g. 44100. */
static ma_atomic_uint32   g_underrunCount;
static ma_atomic_uint32   g_overflowDropCount;

void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount)
{
    const ma_uint8* pInputBytes = (const ma_uint8*)pInput;
    ma_uint32 frameCountInRemaining = frameCount;
    ma_uint32 bytesPerFrameIn = ma_get_bytes_per_frame(pDevice->capture.format, pDevice->capture.channels);
    ma_uint32 rateIn;

    (void)pOutput;

    /*
    Apply any pending drift correction. The converter is only ever touched from the audio
    thread; the controller on the main thread just publishes the desired rate. The resampler
    keeps its filter state across ma_data_converter_set_rate() so this is glitch-free.
    */
    rateIn = ma_atomic_uint32_get(&g_pendingRateIn);
    if (rateIn != 0 && rateIn != g_converter.sampleRateIn) {
        ma_data_converter_set_rate(&g_converter, rateIn, ENCODER_SAMPLE_RATE);
    }

    /*
    Resample this block with the persistent converter. Filter state carries over from the
    previous callback, so there are no discontinuities at block boundaries.
    */
    while (frameCountInRemaining > 0) {
        ma_uint8  convertedFrames[PACKET_FRAMES * 4 * sizeof(ma_int16) * ENCODER_CHANNELS];    /* Large enough for one callback's worth of output. */
        ma_uint64 frameCountIn  = frameCountInRemaining;
        ma_uint64 frameCountOut = sizeof(convertedFrames) / (sizeof(ma_int16) * ENCODER_CHANNELS);

        if (ma_data_converter_process_pcm_frames(&g_converter, pInputBytes, &frameCountIn, convertedFrames, &frameCountOut) != MA_SUCCESS) {
            break;
        }

        pInputBytes           += frameCountIn * bytesPerFrameIn;
        frameCountInRemaining -= (ma_uint32)frameCountIn;

        /* Push the converted frames into the ring buffer for the encoder side to drain. */
        while (frameCountOut > 0) {
            ma_uint32 framesToWrite = (ma_uint32)frameCountOut;
            void* pWritePtr;

            if (ma_pcm_rb_acquire_write(&g_rb, &framesToWrite, &pWritePtr) != MA_SUCCESS || framesToWrite == 0) {
                /* The consumer has stalled. Dropping is the only option here. */
                ma_atomic_uint32_fetch_add(&g_overflowDropCount, (ma_uint32)frameCountOut);
                break;
            }

            MA_COPY_MEMORY(pWritePtr, convertedFrames, framesToWrite * sizeof(ma_int16) * ENCODER_CHANNELS);
            ma_pcm_rb_commit_write(&g_rb, framesToWrite);

            frameCountOut -= framesToWrite;
        }
    }
}

/*
Drift controller. Runs on the encoder side once per packet. Returns the input rate the
converter should use in order to keep the ring buffer centered on RB_TARGET_FRAMES.
*/
static ma_uint32 drift_controller_update(float* pIntegrator, ma_uint32 availableFrames)
{
    ma_int32 error = (ma_int32)availableFrames - (ma_int32)RB_TARGET_FRAMES;
    float correction;
    float integratorLimit = DRIFT_MAX_CORRECTION_HZ / DRIFT_KI;     /* In frame-seconds. */
    ma_int32 rateIn;

    /* Integrate the error (dt = one 10 ms packet per tick) so steady-state drift is fully absorbed. */
    *pIntegrator += (float)error * 0.01f;
    if (*pIntegrator > +integratorLimit) { *pIntegrator = +integratorLimit; }
    if (*pIntegrator < -integratorLimit) { *pIntegrator = -integratorLimit; }

    correction  = DRIFT_KP * (float)error;      /* Proportional: react to the current offset. */
    correction += DRIFT_KI * (*pIntegrator);    /* Integral: soak up the constant clock drift. */

    if (correction > +DRIFT_MAX_CORRECTION_HZ) { correction = +DRIFT_MAX_CORRECTION_HZ; }
    if (correction < -DRIFT_MAX_CORRECTION_HZ) { correction = -DRIFT_MAX_CORRECTION_HZ; }

    rateIn = (ma_int32)g_baseRateIn + (ma_int32)correction;
    if (rateIn < 1) {
        rateIn = 1;
    }

    return (ma_uint32)rateIn;
}

int main(int argc, char** argv)
{
    ma_result result;
    ma_encoder_config encoderConfig;
    ma_encoder encoder;
    ma_device_config deviceConfig;
    ma_device device;
    ma_data_converter_config converterConfig;
    ma_uint32 durationInSeconds;
    ma_uint32 packetCount;
    ma_uint32 i;
    float driftIntegrator = 0;
    ma_timer packetTimer;
    double nextPacketTime;
    static ma_int16 silence[PACKET_FRAMES * ENCODER_CHANNELS];

    if (argc < 2) {
        printf("No output file.\n");
        return -1;
    }

    durationInSeconds = (argc >= 3) ? (ma_uint32)atoi(argv[2]) : 30;
    packetCount = durationInSeconds * 100;  /* 100 packets of 10 ms per second. */

    encoderConfig = ma_encoder_config_init(ma_encoding_format_wav, ENCODER_FORMAT, ENCODER_CHANNELS, ENCODER_SAMPLE_RATE);

    if (ma_encoder_init_file(argv[1], &encoderConfig, &encoder) != MA_SUCCESS) {
        printf("Failed to initialize output file.\n");
        return -1;
    }

    /*
    Open the device at its native rate (sampleRate = 0) so we get raw, unconverted input
    and can own the resampling ourselves.
    */
    deviceConfig = ma_device_config_init(ma_device_type_capture);
    deviceConfig.capture.format             = ENCODER_FORMAT;
    deviceConfig.capture.channels           = ENCODER_CHANNELS;
    deviceConfig.sampleRate                 = 0;    /* Native device rate, e.g. 44100. */
    deviceConfig.periodSizeInMilliseconds   = 10;
    deviceConfig.dataCallback               = data_callback;

    result = ma_device_init(NULL, &deviceConfig, &device);
    if (result != MA_SUCCESS) {
        ma_encoder_uninit(&encoder);
        printf("Failed to initialize capture device.\n");
        return -2;
    }

    g_baseRateIn = device.sampleRate;
    printf("Device native rate: %u Hz, encoding to: %u Hz\n", g_baseRateIn, (ma_uint32)ENCODER_SAMPLE_RATE);

    /*
    One persistent converter for the whole session. allowDynamicSampleRate keeps the
    resampler active even if the rates happen to match, and lets us adjust the rate on
    the fly for drift compensation.
    */
    converterConfig = ma_data_converter_config_init(ENCODER_FORMAT, ENCODER_FORMAT, ENCODER_CHANNELS, ENCODER_CHANNELS, g_baseRateIn, ENCODER_SAMPLE_RATE);
    converterConfig.allowDynamicSampleRate = MA_TRUE;

    result = ma_data_converter_init(&converterConfig, NULL, &g_converter);
    if (result != MA_SUCCESS) {
        ma_device_uninit(&device);
        ma_encoder_uninit(&encoder);
        printf("Failed to initialize data converter.\n");
        return -3;
    }

    result = ma_pcm_rb_init(ENCODER_FORMAT, ENCODER_CHANNELS, RB_CAPACITY_FRAMES, NULL, NULL, &g_rb);
    if (result != MA_SUCCESS) {
        ma_data_converter_uninit(&g_converter, NULL);
        ma_device_uninit(&device);
        ma_encoder_uninit(&encoder);
        printf("Failed to initialize ring buffer.\n");
        return -4;
    }

    /* Pre-center the buffer so the consumer can start immediately and the controller starts at equilibrium. */
    ma_pcm_rb_seek_write(&g_rb, RB_TARGET_FRAMES);

    ma_atomic_uint32_set(&g_pendingRateIn, g_baseRateIn);
    ma_atomic_uint32_set(&g_underrunCount, 0);
    ma_atomic_uint32_set(&g_overflowDropCount, 0);

    result = ma_device_start(&device);
    if (result != MA_SUCCESS) {
        ma_pcm_rb_uninit(&g_rb);
        ma_data_converter_uninit(&g_converter, NULL);
        ma_device_uninit(&device);
        ma_encoder_uninit(&encoder);
        printf("Failed to start device.\n");
        return -5;
    }

    printf("Recording %u seconds...\n", durationInSeconds);

    /*
    Encoder-side loop. This paces itself like a packetizer/encoder would. Each tick it
    drains one packet, feeds the encoder, and updates the drift controller based on the
    ring buffer fill level. Pacing is done against an absolute monotonic clock rather
    than a fixed sleep so per-tick scheduling jitter does not accumulate into a rate
    error that the drift controller would have to fight.
    */
    ma_timer_init(&packetTimer);
    nextPacketTime = 0;

    for (i = 0; i < packetCount; ++i) {
        ma_uint32 available = ma_pcm_rb_available_read(&g_rb);
        ma_uint32 framesRemaining;
        double now;

        /* Publish the corrected rate; the audio thread applies it before the next conversion. */
        ma_atomic_uint32_set(&g_pendingRateIn, drift_controller_update(&driftIntegrator, available));

        framesRemaining = PACKET_FRAMES;
        if (available < PACKET_FRAMES) {
            /* Keep the output timeline intact on underrun. */
            ma_atomic_uint32_fetch_add(&g_underrunCount, 1);
            ma_encoder_write_pcm_frames(&encoder, silence, PACKET_FRAMES, NULL);
        } else {
            while (framesRemaining > 0) {
                ma_uint32 framesToRead = framesRemaining;
                const void* pReadPtr;

                if (ma_pcm_rb_acquire_read(&g_rb, &framesToRead, (void**)&pReadPtr) != MA_SUCCESS || framesToRead == 0) {
                    break;
                }

                ma_encoder_write_pcm_frames(&encoder, pReadPtr, framesToRead, NULL);
                ma_pcm_rb_commit_read(&g_rb, framesToRead);

                framesRemaining -= framesToRead;
            }
        }

        if ((i % 100) == 0) {
            printf("t=%3us  buffered=%4u frames  rateIn=%u Hz\n", i / 100, available, ma_atomic_uint32_get(&g_pendingRateIn));
        }

        nextPacketTime += 0.010;
        now = ma_timer_get_time_in_seconds(&packetTimer);
        if (now < nextPacketTime) {
            ma_sleep((ma_uint32)((nextPacketTime - now) * 1000));
        }
    }

    printf("Underruns: %u, dropped frames due to overflow: %u\n",
        ma_atomic_uint32_get(&g_underrunCount),
        ma_atomic_uint32_get(&g_overflowDropCount));

    ma_device_uninit(&device);
    ma_pcm_rb_uninit(&g_rb);
    ma_data_converter_uninit(&g_converter, NULL);
    ma_encoder_uninit(&encoder);

    return 0;
}
