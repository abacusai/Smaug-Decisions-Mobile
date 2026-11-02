// JNI bridge: ai.abacus.decide.Decider <-> decide_core (the same C core llama_decide.py uses through ctypes).
// Builds for Android (arm64-v8a, x86_64) and for the desktop JVM (host e2e test, see android/run_host_e2e.sh).
#include <jni.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __ANDROID__
#include <android/log.h>
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#else
#define LOGI(...) do { if (getenv("DEC_VERBOSE")) { fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)
#endif

#include "decide_core.h"

#define TAG "decide"

JNIEXPORT jlong JNICALL Java_ai_abacus_decide_Decider_nativeLoad(JNIEnv * env, jclass cls, jstring model, jstring mmproj,
                                                                 jint threads, jint ngl, jint nctx, jint image_max_tokens) {
    (void) cls;
    const char * m = (*env)->GetStringUTFChars(env, model, NULL);
    const char * p = mmproj ? (*env)->GetStringUTFChars(env, mmproj, NULL) : NULL;
    dec_ctx * d = dec_load(m, p, threads, ngl, nctx, image_max_tokens, ngl != 0);
    LOGI("load %s mmproj=%s -> %p", m, p ? p : "-", (void *) d);
    (*env)->ReleaseStringUTFChars(env, model, m);
    if (p) (*env)->ReleaseStringUTFChars(env, mmproj, p);
    return (jlong) (intptr_t) d;
}

JNIEXPORT void JNICALL Java_ai_abacus_decide_Decider_nativeFree(JNIEnv * env, jclass cls, jlong h) {
    (void) env; (void) cls; dec_free((dec_ctx *) (intptr_t) h);
}

JNIEXPORT jint JNICALL Java_ai_abacus_decide_Decider_nativeTokenId(JNIEnv * env, jclass cls, jlong h, jstring s) {
    (void) cls;
    const char * c = (*env)->GetStringUTFChars(env, s, NULL);
    jint id = dec_token_id((dec_ctx *) (intptr_t) h, c);
    (*env)->ReleaseStringUTFChars(env, s, c);
    return id;
}

JNIEXPORT jfloat JNICALL Java_ai_abacus_decide_Decider_nativeSoftcap(JNIEnv * env, jclass cls, jlong h) {
    (void) env; (void) cls; return dec_softcap((dec_ctx *) (intptr_t) h);
}

// prompt: the rendered prompt as UTF-8 bytes (not a jstring: GetStringUTFChars returns *modified* UTF-8, which encodes emoji and
// other supplementary characters as surrogate pairs, so llama.cpp would tokenize different bytes than the Python path).
// media: 0 text (data null), 1 image (byte[] RGB, nx, ny), 2 audio (float[] 16 kHz mono). Returns float[n_ids + 2]:
// the letter logits, then the number of prompt positions and the milliseconds (negative positions = error code).
JNIEXPORT jfloatArray JNICALL Java_ai_abacus_decide_Decider_nativeEval(JNIEnv * env, jclass cls, jlong h, jbyteArray prompt, jint media,
                                                                       jbyteArray rgb, jfloatArray pcm, jint nx, jint ny, jintArray ids) {
    (void) cls;
    dec_ctx * d = (dec_ctx *) (intptr_t) h;
    jsize n_text = (*env)->GetArrayLength(env, prompt);
    char * text = malloc((size_t) n_text + 1);
    (*env)->GetByteArrayRegion(env, prompt, 0, n_text, (jbyte *) text);
    text[n_text] = 0;
    jsize n_ids = (*env)->GetArrayLength(env, ids);
    jint * id = (*env)->GetIntArrayElements(env, ids, NULL);
    float * out = calloc((size_t) n_ids + 2, sizeof(float));
    void * data = NULL;
    if (media == DEC_IMAGE_RGB) {
        if ((int64_t) (*env)->GetArrayLength(env, rgb) != (int64_t) nx * ny * 3) nx = ny = 0;   // size mismatch -> dec_eval error
        data = (*env)->GetByteArrayElements(env, rgb, NULL);
    }
    if (media == DEC_AUDIO_PCM16K) { data = (*env)->GetFloatArrayElements(env, pcm, NULL); nx = (*env)->GetArrayLength(env, pcm); }
    int32_t n = (media == DEC_IMAGE_RGB && nx == 0) ? -9 : dec_eval(d, text, media, data, nx, ny, (const int32_t *) id, n_ids, out);
    out[n_ids] = (float) n;
    out[n_ids + 1] = (float) dec_last_ms(d);
    if (media == DEC_IMAGE_RGB) (*env)->ReleaseByteArrayElements(env, rgb, data, JNI_ABORT);
    if (media == DEC_AUDIO_PCM16K) (*env)->ReleaseFloatArrayElements(env, pcm, data, JNI_ABORT);
    (*env)->ReleaseIntArrayElements(env, ids, id, JNI_ABORT);
    free(text);
    jfloatArray res = (*env)->NewFloatArray(env, n_ids + 2);
    (*env)->SetFloatArrayRegion(env, res, 0, n_ids + 2, out);
    free(out);
    return res;
}
