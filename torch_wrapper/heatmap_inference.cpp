#include "torch_wrapper/heatmap_inference.h"

#include <torch/torch.h>
#include <common_utils/timing/TimerCPU.h>

#include "torch_wrapper/internal/torch_model.h"

namespace torch_wrapper {

	HeatmapInference::HeatmapInference(const std::string& moduleScriptPath) {
		m_model = std::make_unique<TorchModel>(moduleScriptPath);
	}

	HeatmapInference::~HeatmapInference() = default;

	void HeatmapInference::predict(
		const std::vector<float>& anchorIn,
		const std::vector<float>& matchIn,
		int width, int height, int nChannels, int batchSize,
		std::vector<float>& heatmapOut,
		std::vector<float>& occlusionScoreOut,
		std::vector<float>& depthScoreOut
	) {
		TIME_CPU_START(HeatmapInference_predict_initialization);
		torch::NoGradGuard no_grad;

		int nSamples = anchorIn.size() / (width * height * nChannels);
		assert(matchIn.size() == width * height * nChannels, "Match image should be only one.");

		if (nSamples % batchSize != 0) {
			std::cout << "Number of samples needs to be a multiple of batch size!" << std::endl;
			return;
		}

		int nForwardPasses = nSamples / batchSize;

		heatmapOut.clear();
		heatmapOut.resize(nSamples * width * height);

		occlusionScoreOut.clear();
		occlusionScoreOut.resize(nSamples);

		depthScoreOut.clear();
		depthScoreOut.resize(nSamples);

		TIME_CPU_STOP(HeatmapInference_predict_initialization);
		TIME_CPU_START(HeatmapInference_predict_loop);

		for (int i = 0; i < nForwardPasses; i++) {
			TIME_CPU_START(HeatmapInference_predict_preprocess);

			// Initialize inputs.
			int offsetInput = i * batchSize * nChannels * height * width;
			torch::Tensor anchorInput = torch::from_blob((void*)(anchorIn.data() + offsetInput), { batchSize, nChannels, height, width }, at::dtype(at::kFloat)).to(m_model->getDeviceType());
			torch::Tensor matchInput = torch::from_blob((void*)matchIn.data(), { 1, nChannels, height, width }, at::dtype(at::kFloat)).to(m_model->getDeviceType());
		
			std::vector<torch::jit::IValue> inputs;
			inputs.push_back(anchorInput);
			inputs.push_back(matchInput);

			TIME_CPU_STOP(HeatmapInference_predict_preprocess);
			TIME_CPU_START(HeatmapInference_predict_forward);

			// Run the network.
			auto outputs = m_model->forward(inputs);

			TIME_CPU_STOP(HeatmapInference_predict_forward);
			TIME_CPU_START(HeatmapInference_predict_postprocess);
		
			// Copy the outputs.
			torch::Tensor heatmap = outputs.toTuple()->elements()[0].toTensor().to(torch::kCPU);
			torch::Tensor occlusionScore = outputs.toTuple()->elements()[1].toTensor().to(torch::kCPU);
			torch::Tensor depthScore = outputs.toTuple()->elements()[2].toTensor().to(torch::kCPU);

			int offsetHeatmap = i * batchSize * height * width;
			std::copy(heatmap.data_ptr<float>(), heatmap.data_ptr<float>() + batchSize * width * height, heatmapOut.data() + offsetHeatmap);

			int offsetOcclusionScore = i * batchSize;
			std::copy(occlusionScore.data_ptr<float>(), occlusionScore.data_ptr<float>() + batchSize, occlusionScoreOut.data() + offsetOcclusionScore);

			int offsetDepthScore = i * batchSize;
			std::copy(depthScore.data_ptr<float>(), depthScore.data_ptr<float>() + batchSize, depthScoreOut.data() + offsetDepthScore);

			TIME_CPU_STOP(HeatmapInference_predict_postprocess);
		}

		TIME_CPU_STOP(HeatmapInference_predict_loop);
	}

} // namespace torch_wrapper


