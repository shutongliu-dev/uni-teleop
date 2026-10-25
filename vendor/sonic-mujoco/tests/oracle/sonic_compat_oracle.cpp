#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "control_policy.hpp"
#include "math_utils.hpp"
#include "policy_parameters.hpp"

template <typename T>
void Print(const char* name, const T& values) {
  std::cout << name;
  for (const auto value : values) std::cout << ' ' << std::setprecision(17) << value;
  std::cout << '\n';
}

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  std::vector<double> observation;
  observation.reserve(994);
  for (int i = 0; i < 64; ++i) observation.push_back(-0.25 + 0.01 * i);

  std::array<std::array<double, 3>, 10> angular_velocity {};
  std::array<std::array<double, 29>, 10> joint_position {};
  std::array<std::array<double, 29>, 10> joint_velocity {};
  std::array<std::array<double, 29>, 10> last_action {};
  std::array<std::array<double, 3>, 10> gravity {};
  for (int frame = 0; frame < 10; ++frame) {
    angular_velocity[frame] = {0.1 * frame, -0.2 + 0.03 * frame, 0.4 - 0.02 * frame};
    std::array<double, 29> q_hardware {};
    std::array<double, 29> dq_hardware {};
    for (int joint = 0; joint < 29; ++joint) {
      q_hardware[joint] = -0.3 + 0.02 * frame + 0.005 * joint;
      dq_hardware[joint] = 0.15 - 0.01 * frame - 0.003 * joint;
      last_action[frame][joint] = -0.5 + 0.04 * frame + 0.01 * joint;
    }
    for (int joint = 0; joint < 29; ++joint) {
      const int source = mujoco_to_isaaclab[joint];
      joint_position[frame][joint] = q_hardware[source] - default_angles[source];
      joint_velocity[frame][joint] = dq_hardware[source];
    }
    std::array<double, 4> quat = {
        1.0 + 0.01 * frame, 0.02 * frame, -0.015 * frame, 0.01 * (frame + 1)};
    double norm = 0.0;
    for (double value : quat) norm += value * value;
    norm = std::sqrt(norm);
    for (double& value : quat) value /= norm;
    gravity[frame] = quat_rotate_d(quat_conjugate_d(quat), {0.0, 0.0, -1.0});
  }
  for (const auto& frame : angular_velocity) observation.insert(observation.end(), frame.begin(), frame.end());
  for (const auto& frame : joint_position) observation.insert(observation.end(), frame.begin(), frame.end());
  for (const auto& frame : joint_velocity) observation.insert(observation.end(), frame.begin(), frame.end());
  for (const auto& frame : last_action) observation.insert(observation.end(), frame.begin(), frame.end());
  for (const auto& frame : gravity) observation.insert(observation.end(), frame.begin(), frame.end());

  PolicyEngine policy;
  if (!policy.Initialize(argv[1], false)) return 3;
  auto& input = policy.GetInputBuffer();
  for (size_t i = 0; i < observation.size(); ++i)
    input[i] = static_cast<float>(observation[i]);
  if (!policy.Infer()) return 4;
  Print("OBSERVATION", observation);
  Print("ACTION", policy.GetActionBuffer());
}
