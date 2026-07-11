/* parselen: reads an input file and copies its first byte's length worth of bytes.
 * SAFE variant: bounds-checks the length against the 8-byte destination buffer.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 2) return 2;
    FILE *fp = fopen(argv[1], "rb");
    if (!fp) return 2;
    fseek(fp, 0, SEEK_END);
    long n = ftell(fp);
    if (n < 1) { fclose(fp); return 2; }
    fseek(fp, 0, SEEK_SET);
    unsigned char *raw = (unsigned char *)malloc((size_t)n);
    if (!raw) { fclose(fp); return 2; }
    if (fread(raw, 1, (size_t)n, fp) != (size_t)n) { free(raw); fclose(fp); return 2; }
    fclose(fp);
    unsigned int len = raw[0];
    if (len > (unsigned int)(n - 1)) { free(raw); return 3; }
    if (len > 8) { free(raw); return 4; }                 /* SAFE: bounds check */
    char *dst = (char *)malloc(8);
    memcpy(dst, raw + 1, len);
    printf("copied %u bytes\n", len);
    free(dst);
    free(raw);
    return 0;
}
