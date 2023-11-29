#include <stdio.h>
#include "image.h"
#include "timer.h"
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int thread_count;

// Function to perform histogram equalisation on an image
void histogram_equalization(Image *srcImage, Image *destImage) {
 
    // declare 2 arrays for storing histogram values (frequencies) and
    // new gray level values (newly mapped pixel values as per algorithm)
    int hist[256] = { 0 };
    int new_gray_level[256] = { 0 };
    int my_rank = omp_get_thread_num();
 
    // Calculating frequency of occurrence for all pixel values
    #pragma omp parallel 
    {

        int local_hist[256] = { 0 };

        #pragma omp for nowait
        for (int row=0; row<srcImage->height; row++) {
            for (int col=0; col<srcImage->width; col++) {
                int index = Index(col, row, srcImage->width, 1, srcImage->bpp);
                local_hist[srcImage->data[index]]++;
            }      
        }

        #pragma omp critical
        for (int i = 0; i < 256; i++) {
            hist[i] += local_hist[i];
        }
    }
    
  
    // calculating cumulative frequency and new gray levels
    long total = srcImage->height*srcImage->width;
    long curr = 0;
    for (int i=0; i<256; i++) {
        // cumulative frequency
        curr += hist[i];

        // calculating new gray level after multiplying by
        // maximum gray count which is 255 and dividing by
        // total number of pixels
        new_gray_level[i] = round((((float)curr) * 255) / total);
    }

    // performing histogram equalisation by mapping new gray levels
    #pragma omp parallel for
    for (int row=0; row<srcImage->height; row++) {
        for (int pix=0; pix<srcImage->width; pix++) {
            int index = Index(pix, row, srcImage->width, 1, srcImage->bpp);
            destImage->data[index] = new_gray_level[srcImage->data[index]];
        }
    }
 
}

// Usage: Prints usage information for the program
int Usage(){
    printf("Usage: image <filename>\n");
    return -1;
}

// main: expects a filename as argument (can be jpg, png, bmp, tga)
int main(int argc,char** argv) {
    if (argc!=3) return Usage();

    thread_count = strtol(argv[1], NULL, 10);
    omp_set_num_threads(thread_count);

    // Verify the number of threads was set correctly
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        if (thread_id == 0) {
            printf("%d thread(s)\n", omp_get_num_threads());
        }
    }
    
    stbi_set_flip_vertically_on_load(0); 
    
    char* fileName = argv[2];

    Image srcImage;   
    srcImage.data = stbi_load(fileName,&srcImage.width,&srcImage.height,&srcImage.bpp,0);
    if (!srcImage.data){
        printf("Error loading file %s.\n",fileName);
        return -1;
    }

    printf("Image size is %d by %d.\n", srcImage.width, srcImage.height);
#ifdef DEBUG
    printf("Image has %d channels.\n", srcImage.bpp);
#endif

    Image destImage;
    destImage.bpp = srcImage.bpp;
    destImage.height = srcImage.height;
    destImage.width = srcImage.width;
    destImage.data = malloc(sizeof(uint8_t)*destImage.width*destImage.bpp*destImage.height);

    double start, finish;

    #ifdef _OPENMP
        #include <omp.h>
    #endif
    #ifdef _OPENMP
        int my_rank = omp_get_thread_num();
        int thread_count = omp_get_num_threads();
    #else
        int my_rank = 0;
        thread_count = 1;
    #endif

    GET_TIME(start);
    histogram_equalization(&srcImage, &destImage);
    GET_TIME(finish);

    printf("Histogram equalization took %f seconds\n", finish-start);

    stbi_write_png("output.png",destImage.width,destImage.height,destImage.bpp,destImage.data,destImage.bpp*destImage.width);
    stbi_image_free(srcImage.data);
    
    free(destImage.data);
    
    return 0;
} /* compile with cc histogram_equalization.c -lm -o heq */