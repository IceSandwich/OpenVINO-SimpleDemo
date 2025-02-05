#include <iostream>
#include <chrono>
#include <iterator>
#include <functional>
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>

int main() {
	ov::Core core;
	std::cout << "Available devices: ";
	std::vector<std::string> devices = core.get_available_devices();
	std::copy(devices.begin(), devices.end(), std::ostream_iterator<std::string>(std::cout, ", "));
	std::cout << std::endl;
	
	//���onnxԼ27ms
	//ov::CompiledModel compiledModel = core.compile_model("Models/YoloPoseV8/yolov8n-pose.onnx", "AUTO:GPU,-CPU", ov::log::level(ov::log::Level::DEBUG));
	//���Int8Լ27ms���ɱ����FP16�����õ�
	ov::CompiledModel compiledModel = core.compile_model("Models/YoloPoseV8/Int8/yolov8n-pose.xml", "AUTO:GPU,-CPU", ov::log::level(ov::log::Level::DEBUG));
	//���Int8_2���������ˣ���onnxֱ��������ģ�ͣ�Լ53ms
	//ov::CompiledModel compiledModel = core.compile_model("Models/YoloPoseV8/Int8_2/yolov8n-pose.xml", "AUTO:GPU,-CPU", ov::log::level(ov::log::Level::DEBUG));

	ov::InferRequest inferRequest = compiledModel.create_infer_request();

	auto inputPort = compiledModel.input();
	auto inputShape = inputPort.get_shape();
	cv::Size inputSize{ static_cast<int>(inputShape[3]), static_cast<int>(inputShape[2]) };
	std::cout << "Input shape: " << inputSize << std::endl;
	std::cout << "Input type: " << inputPort.get_element_type() << std::endl;

	//cv::Mat img = cv::imread("D:/SRCCODES/datas/3.jpg", cv::ImreadModes::IMREAD_COLOR);
	cv::Mat img{ inputSize, CV_8UC3 };
	cv::resize(img, img, inputSize);
	img.convertTo(img, CV_32F, 255.0);

	ov::Tensor inputTensor{ inputPort.get_element_type(), inputPort.get_shape(), img.data };
	inferRequest.set_input_tensor(inputTensor);

auto tick = std::chrono::high_resolution_clock::now();
	inferRequest.start_async();
	inferRequest.wait();
	auto tock = std::chrono::high_resolution_clock::now();
	auto cost_ms = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(tock - tick);
	//printf("Infer costs %5.f ms\n", cost_ms.count());
	std::cout << "Infer costs " << std::setprecision(5) << cost_ms.count() << " ms" << std::endl;


	auto output = inferRequest.get_tensor("output0");
	std::cout << "Output shape: " << output.get_shape() << std::endl;
	const float *outputBuffer = output.data<const float>();


	return 0;
}
