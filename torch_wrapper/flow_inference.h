#pragma once
#include <vector>
#include <memory>

namespace torch_wrapper {

	class TorchModel;

	class FlowInference {
	public:
		FlowInference(const std::string& moduleScriptPath);

		~FlowInference();

		void predict(
			const std::vector<float>& sourceIn,
			const std::vector<float>& targetIn,
			int width, int height, int nChannels, int batchSize,
			std::vector<float>& flowOut
		);

	private:
		std::unique_ptr<TorchModel> m_model;
	};

} // namespace torch_wrapper