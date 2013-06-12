static inline void SWAP(float &a, float &b)
{
    float tmp = a;
    a = b;
    b = tmp;
}

static inline float _quickselect(float *p, int count, int desired_pos)
{
    if (count == 1) return p[0];

    float pivot = p[count - 1];
    int lo_dest = 0;
    for (int i = 0; i < count - 1; i++) {
        if (p[i] <= pivot) {
            SWAP(p[i], p[lo_dest]);
            lo_dest += 1;
        }
    }
    // move pivot to the right place
    SWAP(p[count - 1], p[lo_dest]);

    if (desired_pos == lo_dest) return pivot;
    if (lo_dest > desired_pos) return _quickselect(p, lo_dest, desired_pos);
    return _quickselect(p + lo_dest + 1, count - lo_dest - 1, desired_pos - lo_dest - 1);
}

static inline float quickmedian(const float *p, int count)
{
    float *temp = new float[count];
    memcpy((void *) temp, (const void *) p, count * sizeof(float));
	float result = _quickselect(temp, count, count/2);
	delete[] temp;
	return result;
}
