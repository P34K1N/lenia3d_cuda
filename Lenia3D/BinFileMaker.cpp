#include <fstream>

int main() {
	int size = 64;
	int area = size * size;
	int volume = area * size;
	float * data = (float*)calloc(volume, sizeof(*data));

	int mid = size / 2;
	data[mid * area + mid * size + mid - 1] = 1.0;
	data[mid * area + (mid + 1) * size + mid] = 1.0;
	data[mid * area + (mid - 1) * size + mid] = 1.0;
	data[mid * area + (mid + 1) * size + mid + 1] = 1.0;
	data[mid * area + (mid + 1) * size + mid - 1] = 1.0;

	std::ofstream output(std::string("init.bin"), std::ios::binary);
	output.write((char*)data, volume * sizeof(*data));
}