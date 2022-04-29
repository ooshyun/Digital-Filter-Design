#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
// Reference.
// https://github.com/scipy/scipy/blob/master/scipy/signal/_sosfilt.pyx
// https://github.com/scipy/scipy/blob/master/scipy/signal/_lfilter.c.in

static void _sosfilt(float *sos, int numsection, float *x, float *y, int numsignal, int numsample, float *zi)
{
	// Apply bilinear transform filter to frame
	// sos = b0, b1, b2, a0, a1, a2 , b0, b1, b2, a0, a1, a2], ...
	// n_sections = number of sections in filter
	// n_signal = the number of frames
	// n_samples = the size of frame
	// zi
	// zi_slice

	// TODO: check if sos is valid
	int n_signals = numsignal;
	int n_samples = numsample;
	int n_sections = numsection;
	int i, n, s;
	float x_new, x_cur;
	
	// even: feedforward, odd: feedback
	size_t size_slice = sizeof(float)*n_samples*n_sections;
	float* zi_slice = (float *)malloc(size_slice); 
	memset(zi_slice, 0, size_slice);


	for(i=0; i<n_signals; i++)
	{
		memcpy(zi_slice, zi+i*size_slice, size_slice);

		for(n=0; n<n_samples; n++)
		{
			int idx0 = i*n_samples;
			x_cur = x[idx0+n];
			for(s=0; s<n_sections; s++)
			{
				// printf("%d, x[%d]=%f, %d, ", i, n, x_cur, s);

				// current
				x_new = sos[s*6+0] * x_cur + zi_slice[s*2+0];

				// middle					
				zi_slice[s*2+0] = sos[s*6+1] * x_cur - sos[s*6+4] * x_new 
								+ zi_slice[s*2+1];
				// last
				zi_slice[s*2+1] = sos[s*6+2] * x_cur - sos[s*6+5] * x_new;					
				x_cur = x_new;

				// printf("%f \n", x_new);
			}
			y[idx0+n] = x_cur;
		}
		memset(zi_slice, 0, size_slice);
	}
	free(zi_slice);
}


static void _sosfilt_direct(float *sos, int numsection, float *x, int numsignal, int numsample, float *zi)
{
	// Apply bilinear transform filter to frame
	// sos = b0, b1, b2, a0, a1, a2 , b0, b1, b2, a0, a1, a2], ...
	// n_sections = number of sections in filter
	// n_signal = the number of frames
	// n_samples = the size of frame
	// zi
	// zi_slice

	// TODO: check if sos is valid
	int n_signals = numsignal;
	int n_samples = numsample;
	int n_sections = numsection;
	int i, n, s;
	float x_new, x_cur;
	
	// even: feedforward, odd: feedback
	size_t size_slice = sizeof(float)*n_samples*n_sections;
	float* zi_slice = (float *)malloc(size_slice); 
	memset(zi_slice, 0, size_slice);


	for(i=0; i<n_signals; i++)
	{
		memcpy(zi_slice, zi+i*size_slice, size_slice);

		for(n=0; n<n_samples; n++)
		{
			int idx0 = i*n_samples;
			x_cur = x[idx0+n];
			for(s=0; s<n_sections; s++)
			{
				// printf("%d, x[%d]=%f, %d, ", i, n, x_cur, s);

				// current
				x_new = sos[s*6+0] * x_cur + zi_slice[s*2+0];

				// middle					
				zi_slice[s*2+0] = sos[s*6+1] * x_cur - sos[s*6+4] * x_new 
								+ zi_slice[s*2+1];
				// last
				zi_slice[s*2+1] = sos[s*6+2] * x_cur - sos[s*6+5] * x_new;					
				x_cur = x_new;

				// printf("%f \n", x_new);
			}
			x[idx0+n] = x_cur;
		}
		memset(zi_slice, 0, size_slice);
	}
	free(zi_slice);
}


static void _filt_float(float *b, float *a, float *x, float *y, float *Z, int len_b, uint32_t len_x, int stride_x, int stride_y)
{
	float *ptr_x = x, *ptr_y = y;
	float *ptr_Z;
	float *ptr_b = (float*)b;
	float *ptr_a = (float*)a;
	float *xn, *yn;
	const float a0 = *((float *)a);
	int n;
	uint32_t k;

	/* normalize the filter coefs only once. */
	for (n = 0; n < len_b; ++n) {
		ptr_b[n] /= a0;
		ptr_a[n] /= a0;
	}

	for (k = 0; k < len_x; k++) {
		ptr_b = (float *)b;   /* Reset a and b pointers */
		ptr_a = (float *)a;
		xn = (float *)ptr_x;
		yn = (float *)ptr_y;
				
		// 1-stage filter
		if (len_b > 1) {
			// IIR filter
			ptr_Z = ((float *)Z);
			*yn = *ptr_Z + *ptr_b  * *xn;   /* Calculate first delay (output) */
			ptr_b++;
			ptr_a++;
			/* Fill in middle delays */
			for (n = 0; n < len_b - 2; n++) {
				*ptr_Z =
					*xn * (*ptr_b) - *yn * (*ptr_a) + ptr_Z[1];
				ptr_b++;
				ptr_a++;
				ptr_Z++;
			}
			/* Calculate last delay */
			*ptr_Z = *xn * (*ptr_b) - *yn * (*ptr_a);
		}
		else {
			// FIR filter
			*yn = *xn * (*ptr_b);
		}
			
		ptr_y += stride_y;      /* Move to next input/output point */
		ptr_x += stride_x;
	}
}

static void _filt_float_direct(float *b, float *a, float *x, float *Z, int len_b, uint32_t len_x, int stride_x)
{
	float *ptr_x = x;
	float *ptr_Z;
	float *ptr_b = (float*)b;
	float *ptr_a = (float*)a;
	float *xn;
	const float a0 = *((float *)a);
	int n;
	uint32_t k;

	float x_cur, x_new;

	/* normalize the filter coefs only once. */
	for (n = 0; n < len_b; ++n) {
		ptr_b[n] /= a0;
		ptr_a[n] /= a0;
	}

	for (k = 0; k < len_x; k++) {
		ptr_b = (float *)b;   /* Reset a and b pointers */
		ptr_a = (float *)a;
		xn = (float *)ptr_x;

		x_cur = *xn;
		// 1-stage filter
		if (len_b > 1) {
			// IIR filter
			ptr_Z = ((float *)Z);

			/* Calculate first delay (output) */
			x_new = *ptr_Z + *ptr_b  * x_cur;   
			ptr_b++;
			ptr_a++;
			
			/* Fill in middle delays */
			for (n = 0; n < len_b - 2; n++) {
				*ptr_Z =
					x_cur * (*ptr_b) - x_new * (*ptr_a) + ptr_Z[1];
				ptr_b++;
				ptr_a++;
				ptr_Z++;
			}
			/* Calculate last delay */
			*ptr_Z = x_cur * (*ptr_b) - x_new * (*ptr_a);
		}
		else {
			// FIR filter
			x_new = x_cur * (*ptr_b);
		}
		// update to intermidiate
		*xn = x_new;
		
		/* Move to next input point */
		ptr_x += stride_x;		
	}
}

int main()
{
	float b[3] = { 0.00013651, 0.00027302, 0.00013651 };
	float a[3] = { 1., -1.96668139 , 0.96722743 };
	
	float x_linear[256] = {0};
	float y_linear[256] = {0};


	// len of delay is len of coff -1
	float delay_linear[2] = {0};
	
	
	float sos[6] = { 0.00013651, 0.00027302, 0.00013651, 1., -1.96668139 , 0.96722743 };
	float x_bilinear[256] = {0};
	float delay_bilnear[2] = {0};
	float y_bilinear[256] = {0};
		

	clock_t start, end;

	double total_time = 0;
	for (int i = 0; i < 1; i++)
	{
		for (int j = 0; j < 256; j++)
		{
			x_linear[j] = (float)j;
			x_bilinear[j] = (float)j;
		}
		memset(y_linear, 0, sizeof(y_linear));
		memset(y_bilinear, 0, sizeof(y_bilinear));

		start = clock();

		/* linear filter */
		memset(delay_linear, 0, sizeof(delay_linear));
		_filt_float(b, a, &x_linear[0], &y_linear[0], delay_linear, 3, 256, 1, 1);

		/* linear filter with no y */
		memset(delay_linear, 0, sizeof(delay_linear));
		_filt_float_direct(b, a, &x_linear[0], delay_linear, 3, 256, 1);

		/* bilnear filter with no y*/
		memset(delay_bilnear, 0, sizeof(delay_bilnear));
		_sosfilt(sos, 1, &x_bilinear[0], &y_bilinear[0], 1, 256, delay_bilnear);

		/* bilnear filter with no y*/
		memset(delay_bilnear, 0, sizeof(delay_bilnear));
		_sosfilt_direct(sos, 1, &x_bilinear[0], 1, 256, delay_bilnear);

		end = clock();

		// printf("-----\ntest result\n-----\n");
		// printf("%f\n",(double)(end - start)/CLOCKS_PER_SEC);
		total_time += (double)(end - start)/CLOCKS_PER_SEC;

		for (int i = 0; i < 256; i++)
		{
			printf("%f, ", y_linear[i]);
			printf("%f, ", x_linear[i]);
			printf("%f, ", y_bilinear[i]);
			printf("%f \n", x_bilinear[i]);
		}	
	}

	printf("\n-----------\ntotal time : %f\n-------------\n", total_time/100);
	// getchar();
}


