#include <fstream>
#include <iostream>

int main() {
	std::cout << "asdas" << std::endl;
	srand(10);
	int size = 64;
	int area = size * size;
	int volume = area * size;
	float * data = (float*)calloc(volume, sizeof(*data));
	std::cout << "asdas" << std::endl;

	// int mid = size / 2;
	// data[mid * area + mid * size + mid - 1] = 1.0;
	// data[mid * area + (mid + 1) * size + mid] = 1.0;
	// data[mid * area + (mid - 1) * size + mid] = 1.0;
	// data[mid * area + (mid + 1) * size + mid + 1] = 1.0;
	// data[mid * area + (mid + 1) * size + mid - 1] = 1.0;
	for (int i = 0; i < volume; i++) {
		data[i] = float(rand()) / RAND_MAX;
	}

	std::cout << "asdas" << std::endl;

	std::ofstream output(std::string("init.bin"), std::ios::binary);
	output.write((char*)data, volume * sizeof(*data));
	std::cout << "asdas" << std::endl;
}