#pragma once
#include <vector>
#include <torch/torch.h>
#include <torch/script.h>

namespace torch_wrapper {

	class TorchModel {
	public:
		TorchModel(const std::string& moduleScriptPath) {
			// Network optimization.
			at::globalContext().setBenchmarkCuDNN(true);

			// Load model.
			m_model = torch::jit::load(moduleScriptPath);

			if (torch::cuda::is_available()) {
				std::cout << "CUDA available! Inference on GPU!" << std::endl;
				m_deviceType = torch::kCUDA;
				m_model.to(m_deviceType);
			}
			else {
				std::cout << "Inference on CPU!" << std::endl;
				m_deviceType = torch::kCPU;
				m_model.to(m_deviceType);
			}
			if (!torch::cuda::cudnn_is_available()) {
				std::cout << "CuDNN not available. Memory will be limited." << std::endl;
			}

			torch::Device device(m_deviceType);
			m_model.to(device);
		}

		template<typename InputType>
		auto forward(InputType&& input) {
			return m_model.forward(input);
		}

		const torch::DeviceType& getDeviceType() {
			return m_deviceType;
		}

	private:
		torch::jit::script::Module m_model;
		torch::DeviceType m_deviceType;
	};

} // namespace torch_wrapper