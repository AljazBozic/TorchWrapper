#include "torch_wrapper/flow_inference.h"

#include <torch/torch.h>
#include <common_utils/timing/TimerCPU.h>

#include "torch_wrapper/internal/torch_model.h"

namespace torch_wrapper {

	FlowInference::FlowInference(const std::string& moduleScriptPath) {
		m_model = std::make_unique<TorchModel>(moduleScriptPath);
	}

	FlowInference::~FlowInference() = default;

	void FlowInference::predict(
		const std::vector<float>& sourceIn,
		const std::vector<float>& targetIn,
		int width, int height, int nChannels, int batchSize,
		std::vector<float>& flowOut
	) {
		TIME_CPU_START(FlowInference_predict_initialization);
		torch::NoGradGuard no_grad;

		int nSamples = sourceIn.size() / (width * height * nChannels);
		assert(targetIn.size() == sourceIn.size(), "Target image dimension should match source image dimension.");

		if (nSamples % batchSize != 0) {
			std::cout << "Number of samples needs to be a multiple of batch size!" << std::endl;
			return;
		}

		int nForwardPasses = nSamples / batchSize;

		int flowDim = 2;

		flowOut.clear();
		flowOut.resize(nSamples * flowDim * width * height);

		TIME_CPU_STOP(FlowInference_predict_initialization);
		TIME_CPU_START(FlowInference_predict_loop);

		for (int i = 0; i < nForwardPasses; i++) {
			TIME_CPU_START(FlowInference_predict_preprocess);

			// Initialize inputs.
			int offsetInput = i * batchSize * nChannels * height * width;
			torch::Tensor sourceInput = torch::from_blob((void*)(sourceIn.data() + offsetInput), { batchSize, nChannels, height, width }, at::dtype(at::kFloat)).to(m_model->getDeviceType());
			torch::Tensor targetInput = torch::from_blob((void*)(targetIn.data() + offsetInput), { batchSize, nChannels, height, width }, at::dtype(at::kFloat)).to(m_model->getDeviceType());
		
			std::vector<torch::jit::IValue> inputs;
			inputs.push_back(sourceInput);
			inputs.push_back(targetInput);

			TIME_CPU_STOP(FlowInference_predict_preprocess);
			TIME_CPU_START(FlowInference_predict_forward);

			// Run the network.
			auto outputs = m_model->forward(inputs);

			TIME_CPU_STOP(FlowInference_predict_forward);
			TIME_CPU_START(FlowInference_predict_postprocess);
		
			// Copy the outputs.
			//torch::Tensor flow = outputs.toTuple()->elements()[0].toTensor().to(torch::kCPU);
			torch::Tensor flow = outputs.toTensor().to(torch::kCPU);

			int offsetFlow = i * batchSize * height * width * flowDim;
			std::copy(flow.data_ptr<float>(), flow.data_ptr<float>() + batchSize * width * height * flowDim, flowOut.data() + offsetFlow);

			TIME_CPU_STOP(FlowInference_predict_postprocess);
		}

		TIME_CPU_STOP(FlowInference_predict_loop);
	}

} // namespace torch_wrapper


