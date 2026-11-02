// decide_replay: a tiny CLI over decide_core for checking a cross-compiled build (e.g. the Android arm64 build under
// qemu-aarch64, see qemu_arm64_check.py) against the Python path on the same inputs. Prints "positions ms z0 z1 ...".
// usage: decide_replay <model.gguf> <mmproj.gguf|-> <threads> <prompt.txt> <n_options> [text|image|audio] [media.raw] [nx] [ny]
//   image: media.raw = nx*ny*3 RGB bytes (already on the HF grid); audio: media.raw = float32 16 kHz mono samples.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "decide_core.h"

static void * slurp(const char * path, long * n) {
    FILE * f = fopen(path, "rb");
    if (!f) { perror(path); exit(2); }
    fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
    char * b = malloc((size_t) *n + 1);
    if (fread(b, 1, (size_t) *n, f) != (size_t) *n) { perror(path); exit(2); }
    b[*n] = 0; fclose(f);
    return b;
}

int main(int argc, char ** argv) {
    if (argc < 6) { fprintf(stderr, "usage: %s model mmproj|- threads prompt.txt n_options [text|image|audio] [media.raw] [nx] [ny]\n", argv[0]); return 2; }
    const char * mm = strcmp(argv[2], "-") == 0 ? NULL : argv[2];
    int threads = atoi(argv[3]), n_opt = atoi(argv[5]);
    long np = 0; char * prompt = slurp(argv[4], &np);
    const char * kind = argc > 6 ? argv[6] : "text";
    int media = strcmp(kind, "image") == 0 ? DEC_IMAGE_RGB : strcmp(kind, "audio") == 0 ? DEC_AUDIO_PCM16K : DEC_TEXT;
    long nm = 0; void * data = media != DEC_TEXT ? slurp(argv[7], &nm) : NULL;
    int nx = media == DEC_IMAGE_RGB ? atoi(argv[8]) : media == DEC_AUDIO_PCM16K ? (int) (nm / 4) : 0;
    int ny = media == DEC_IMAGE_RGB ? atoi(argv[9]) : 0;
    dec_ctx * d = dec_load(argv[1], mm, threads, 0, 4096, 560, 0);
    if (!d) { fprintf(stderr, "load failed\n"); return 1; }
    const char * L = "ABCDEFGHIJKLMNOPQRST";
    int32_t ids[20]; float out[20];
    for (int i = 0; i < n_opt; i++) { char s[2] = { L[i], 0 }; ids[i] = dec_token_id(d, s); }
    int32_t n = dec_eval(d, prompt, media, data, nx, ny, ids, n_opt, out);
    printf("%d %.0f", n, dec_last_ms(d));
    for (int i = 0; i < n_opt; i++) printf(" %.6f", out[i]);
    printf("\n");
    dec_free(d);
    return n > 0 ? 0 : 1;
}
