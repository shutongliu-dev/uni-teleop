#include <iomanip>
#include <iostream>
#include <vector>

#include "encoder.hpp"

template <typename T>
void Print(const char* name, const T& values) {
  std::cout << name;
  for (const auto value : values)
    std::cout << ' ' << std::setprecision(17) << value;
  std::cout << '\n';
}

int main(int argc, char** argv) {
  if (argc != 2) return 2;

  std::vector<double> observation(1762, 0.0);
  observation[0] = 2.0;
  for (int frame = 0; frame < 10; ++frame) {
    for (int joint = 0; joint < 24; ++joint) {
      for (int xyz = 0; xyz < 3; ++xyz) {
        observation[922 + frame * 72 + joint * 3 + xyz] = frame;
      }
    }
    const double orientation[] = {1.0, 0.0, 0.0, 1.0, 0.0, 0.0};
    for (int i = 0; i < 6; ++i)
      observation[1642 + frame * 6 + i] = orientation[i];
    const int wrists[] = {23, 24, 25, 26, 27, 28};
    for (int i = 0; i < 6; ++i)
      observation[1702 + frame * 6 + i] = frame * 100 + wrists[i];
  }

  EncoderEngine encoder;
  if (!encoder.Initialize(argv[1], false)) return 3;
  auto& input = encoder.GetInputBuffer();
  for (size_t i = 0; i < observation.size(); ++i)
    input[i] = static_cast<float>(observation[i]);
  if (!encoder.Encode()) return 4;

  Print("OBSERVATION", observation);
  Print("TOKEN", encoder.GetTokenBuffer());
}
