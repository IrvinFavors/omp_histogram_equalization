# omp_histogram_equalization

Parallelization of a histogram equalization algorithm. Histogram equalization is a method used in image processing to better distribute the intensities of a grayscale image to utilize the full range of intensity values. This will allow areas of low local contrast to gain higher contrast and can lead to enhanced details in photographs that are over- or underexposed.

## Compile: cc -DEBUG omp_histogram_equalization.c -lm -o heq -fopenmp
## Run: ./heq 4 bridge.png
