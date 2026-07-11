/* tokread: reads a JSON-like input and expects a "value":"..." field.
 * SAFE variant: only frees the value buffer at end of program.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *g_value = NULL;

static void set_value(const char *s, size_t n) {
    free(g_value);
    g_value = (char *)malloc(n + 1);
    memcpy(g_value, s, n);
    g_value[n] = '\0';
}

int main(int argc, char **argv) {
    if (argc < 2) return 2;
    FILE *fp = fopen(argv[1], "rb");
    if (!fp) return 2;
    fseek(fp, 0, SEEK_END);
    long n = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (n < 0) { fclose(fp); return 2; }
    char *buf = (char *)malloc((size_t)n + 1);
    if (fread(buf, 1, (size_t)n, fp) != (size_t)n) { free(buf); fclose(fp); return 2; }
    buf[n] = '\0';
    fclose(fp);

    const char *needle = "\"value\":\"";
    char *p = strstr(buf, needle);
    if (p) {
        p += strlen(needle);
        char *end = strchr(p, '"');
        if (end) set_value(p, (size_t)(end - p));
    }

    /* Command: RESET; PRINT sequence. */
    if (strstr(buf, "RESET")) {
        /* SAFE: reset does NOT free the value here. */
    }
    if (strstr(buf, "PRINT")) {
        if (g_value) printf("value=%s\n", g_value);
    }

    free(g_value);
    g_value = NULL;
    free(buf);
    return 0;
}
