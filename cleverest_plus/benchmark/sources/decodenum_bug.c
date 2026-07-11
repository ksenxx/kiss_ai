/* decodenum: reads two decimal integers separated by whitespace ("count" and
 * "stride"), allocates an array of "count" ints, and writes stride*i into
 * index i for i in [0, count).
 *
 * BUG variant: the write loop uses stride+i as the destination index, which
 * escapes the allocated region when stride is positive. Safe variant writes
 * to index i only.
 */
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 2) return 2;
    FILE *fp = fopen(argv[1], "rb");
    if (!fp) return 2;
    long count = 0, stride = 0;
    if (fscanf(fp, "%ld %ld", &count, &stride) != 2) { fclose(fp); return 2; }
    fclose(fp);
    if (count <= 0 || count > 64) return 3;
    if (stride < -1000 || stride > 1000) return 3;
    int *arr = (int *)malloc((size_t)count * sizeof(int));
    if (!arr) return 2;
    for (long i = 0; i < count; ++i) arr[i + stride] = (int)(stride * i); /* BUG */
    printf("last=%d\n", arr[count - 1]);
    free(arr);
    return 0;
}
