/* test_quickmedian.c - */

#include <iostream>
#include "quickmedian.h"

int main(int argc, char *argv[])
{
    float testvec[1000];
    while (1) {
        int numgen = rand() % 1000;
        for (int i = 0; i < numgen; i++)
            testvec[i] = (float)rand();
        float med = quickmedian(testvec, numgen);
        //cout << med << endl;
        int numlo = 0;
        for (int i = 0; i < numgen; i++)
            if (testvec[i] < med)
                numlo++;
        //cout << numlo << " " << numgen / 2 << endl;
    }
}
