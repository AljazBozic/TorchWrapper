#include <torch/torch.h>
#include <iostream>

#include <common_utils/timing/TimerCPU.h>

#include "torch_wrapper/heatmap_inference.h"
#include "torch_wrapper/flow_inference.h"

using namespace torch_wrapper;

void heatmapPredictionTest() {
	std::string modelPath = "C:/Workspace/BaseDeform/data/heatmap_prediction.pt";
	int numSamples = 20;
	int width = 224;
	int height = 224;
	int nChannels = 6;
	int batchSize = 10;

	HeatmapInference heatmapInference{ modelPath };

	std::vector<float> anchorInput, matchInput, heatmapOutput, occlusionScoreOutput, depthScoreOutput;
	anchorInput.resize(numSamples * width * height * nChannels, 0.f);
	matchInput.resize(width * height * nChannels, 0.f);

	heatmapInference.predict(
		anchorInput, matchInput,
		width, height, nChannels, batchSize,
		heatmapOutput, occlusionScoreOutput, depthScoreOutput
	);
}

void flowPredictionTest() {
	std::string modelPath = "C:/Workspace/BaseDeform/data/flow_prediction.pt";
	int numSamples = 1;//20;
	int width = 640;
	int height = 480;
	int nChannels = 6;
	int batchSize = 1;//10;

	FlowInference flowInference{ modelPath };

	std::vector<float> sourceInput, targetInput, flowOutput;
	sourceInput.resize(numSamples * width * height * nChannels, 0.f);
	targetInput.resize(numSamples * width * height * nChannels, 0.f);

	flowInference.predict(
		sourceInput, targetInput,
		width, height, nChannels, batchSize,
		flowOutput
	);
}

void heatmapPredictionBenchmark() {
	std::string modelPath = "C:/Workspace/BaseDeform/data/heatmap_prediction.pt";
	int width = 224;
	int height = 224;
	int nChannels = 6;
	int batchSize = 96;
	int numSamples = 3 * batchSize;
	int nIterations = 10;

	HeatmapInference heatmapInference{ modelPath };

	std::vector<float> anchorInput, matchInput, heatmapOutput, occlusionScoreOutput, depthScoreOutput;
	anchorInput.resize(numSamples * width * height * nChannels, 0.f);
	matchInput.resize(width * height * nChannels, 0.f);

	TIME_CPU_START(Benchmark);

	for (int i = 0; i < nIterations; i++) {
		heatmapInference.predict(
			anchorInput, matchInput,
			width, height, nChannels, batchSize,
			heatmapOutput, occlusionScoreOutput, depthScoreOutput
		);
	}

	TIME_CPU_STOP(Benchmark);
}

void flowPredictionBenchmark() {
	std::string modelPath = "C:/Workspace/BaseDeform/data/flow_prediction.pt";
	int numSamples = 20;
	int width = 640;
	int height = 480;
	int nChannels = 6;
	int batchSize = 10;
	int nIterations = 10;

	FlowInference flowInference{ modelPath };

	std::vector<float> sourceInput, targetInput, flowOutput;
	sourceInput.resize(numSamples * width * height * nChannels, 0.f);
	targetInput.resize(numSamples * width * height * nChannels, 0.f);

	TIME_CPU_START(Benchmark);

	for (int i = 0; i < nIterations; i++) {
		flowInference.predict(
			sourceInput, targetInput,
			width, height, nChannels, batchSize,
			flowOutput
		);
	}

	TIME_CPU_STOP(Benchmark);
}

int main() {
	//flowPredictionTest();
	flowPredictionBenchmark();

	return 0;
}