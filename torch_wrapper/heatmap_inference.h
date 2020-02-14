#pragma once
#include <vector>
#include <memory>

namespace torch_wrapper {

	class TorchModel;

	class HeatmapInference {
	public:
		HeatmapInference(const std::string& moduleScriptPath);

		~HeatmapInference();

		void predict(
			const std::vector<float>& anchorBatchIn,
			const std::vector<float>& matchIn,
			int width, int height, int nChannels, int batchSize,
			std::vector<float>& heatmapOut,
			std::vector<float>& occlusionScoreOut,
			std::vector<float>& depthScoreOut
		);

	private:
		std::unique_ptr<TorchModel> m_model;
	};

} // namespace torch_wrapper