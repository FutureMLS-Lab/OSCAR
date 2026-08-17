import Foundation

struct Model: Identifiable {
    var id = UUID()
    var name: String
    var url: String
    var filename: String
    var status: String?
}

@MainActor
class LlamaState: ObservableObject {
    @Published var messageLog = ""
    @Published var cacheCleared = false
    @Published var downloadedModels: [Model] = []
    @Published var undownloadedModels: [Model] = []
    @Published var isLoadingModel = false
    @Published var isGenerating = false
    @Published var loadingStatus = "No model loaded"
    @Published var loadedModelName: String?
    let NS_PER_S = 1_000_000_000.0

    private var llamaContext: LlamaContext?
    private var stopRequested = false
    private var defaultModelUrl: URL? {
        Bundle.main.url(forResource: "ggml-model", withExtension: "gguf", subdirectory: "models")
        // Bundle.main.url(forResource: "llama-2-7b-chat", withExtension: "Q2_K.gguf", subdirectory: "models")
    }

    init() {
        loadModelsFromDisk()
        loadDefaultModels()
        runSimulatorSelfTestIfRequested()
    }

    private func loadModelsFromDisk() {
        do {
            let documentsURL = getDocumentsDirectory()
            let modelURLs = try FileManager.default.contentsOfDirectory(at: documentsURL, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants])
            for modelURL in modelURLs {
                guard modelURL.pathExtension.lowercased() == "gguf" else {
                    continue
                }

                let modelName = modelURL.deletingPathExtension().lastPathComponent
                downloadedModels.append(Model(name: modelName, url: "", filename: modelURL.lastPathComponent, status: "downloaded"))
            }
        } catch {
            print("Error loading models from disk: \(error)")
        }
    }

    private func loadDefaultModels() {
        if let defaultModelUrl {
            Task {
                _ = await loadModel(modelUrl: defaultModelUrl)
            }
        } else {
            messageLog += "OSCAR runtime ready. Import or download a GGUF model to begin.\n"
        }

        for model in defaultModels {
            let fileURL = getDocumentsDirectory().appendingPathComponent(model.filename)
            if FileManager.default.fileExists(atPath: fileURL.path) {

            } else {
                var undownloadedModel = model
                undownloadedModel.status = "download"
                undownloadedModels.append(undownloadedModel)
            }
        }
    }

    private func runSimulatorSelfTestIfRequested() {
        guard let filename = ProcessInfo.processInfo.environment["OSCAR_AUTOTEST_MODEL_FILENAME"] else {
            return
        }

        Task {
            let modelURL = getDocumentsDirectory().appendingPathComponent(filename)
            resetSelfTestReport()
            messageLog += "\n[SelfTest] Loading \(filename)\n"
            recordSelfTest("[SelfTest] Loading \(filename)")
            print("[SelfTest] Loading \(filename)")
            let didLoad = await loadModel(modelUrl: modelURL)
            guard didLoad else {
                messageLog += "\n[SelfTest] FAILED\n"
                recordSelfTest("[SelfTest] FAILED")
                print("[SelfTest] FAILED")
                return
            }

            let prompt = ProcessInfo.processInfo.environment["OSCAR_AUTOTEST_PROMPT"] ?? ""
            let completionTokens = Int(ProcessInfo.processInfo.environment["OSCAR_AUTOTEST_COMPLETION_TOKENS"] ?? "") ?? 8
            await runQuickCompletionSelfTest(prompt: prompt, maxTokens: completionTokens)
            let benchPromptTokens = Int(ProcessInfo.processInfo.environment["OSCAR_AUTOTEST_BENCH_PP"] ?? "") ?? 8
            let benchGenerationTokens = Int(ProcessInfo.processInfo.environment["OSCAR_AUTOTEST_BENCH_TG"] ?? "") ?? 4
            await runQuickBenchmarkSelfTest(pp: benchPromptTokens, tg: benchGenerationTokens)
            messageLog += "\n[SelfTest] COMPLETE\n"
            recordSelfTest("[SelfTest] COMPLETE")
            print("[SelfTest] COMPLETE")
        }
    }

    private func selfTestReportURL() -> URL {
        getDocumentsDirectory().appendingPathComponent("oscar-selftest.log")
    }

    private func resetSelfTestReport() {
        try? FileManager.default.removeItem(at: selfTestReportURL())
    }

    private func recordSelfTest(_ line: String) {
        guard ProcessInfo.processInfo.environment["OSCAR_AUTOTEST_MODEL_FILENAME"] != nil else {
            return
        }

        let data = Data((line + "\n").utf8)
        let url = selfTestReportURL()

        if FileManager.default.fileExists(atPath: url.path) {
            if let handle = try? FileHandle(forWritingTo: url) {
                _ = try? handle.seekToEnd()
                try? handle.write(contentsOf: data)
                try? handle.close()
            }
        } else {
            try? data.write(to: url)
        }
    }

    func getDocumentsDirectory() -> URL {
        let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
        return paths[0]
    }
    private let defaultModels: [Model] = [
        Model(
            name: "OSCAR Gemma-4-12B-it (Q4_K_M rot-kv, 7.4 GiB)",
            url: "https://huggingface.co/Zhongzhu/OSCAR-LLAMACPP-Gemma-4-12B-it-INT2-KV/resolve/main/q4km-rot-kv/gemma-4-12b-it-rot-kv.gguf?download=true",
            filename: "gemma-4-12b-it-rot-kv.gguf",
            status: "download"
        ),
        Model(name: "TinyLlama-1.1B (Q4_0, 0.6 GiB)",url: "https://huggingface.co/TheBloke/TinyLlama-1.1B-1T-OpenOrca-GGUF/resolve/main/tinyllama-1.1b-1t-openorca.Q4_0.gguf?download=true",filename: "tinyllama-1.1b-1t-openorca.Q4_0.gguf", status: "download"),
        Model(
            name: "TinyLlama-1.1B Chat (Q8_0, 1.1 GiB)",
            url: "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf?download=true",
            filename: "tinyllama-1.1b-chat-v1.0.Q8_0.gguf", status: "download"
        ),

        Model(
            name: "TinyLlama-1.1B (F16, 2.2 GiB)",
            url: "https://huggingface.co/ggml-org/models/resolve/main/tinyllama-1.1b/ggml-model-f16.gguf?download=true",
            filename: "tinyllama-1.1b-f16.gguf", status: "download"
        ),

        Model(
            name: "Phi-2.7B (Q4_0, 1.6 GiB)",
            url: "https://huggingface.co/ggml-org/models/resolve/main/phi-2/ggml-model-q4_0.gguf?download=true",
            filename: "phi-2-q4_0.gguf", status: "download"
        ),

        Model(
            name: "Phi-2.7B (Q8_0, 2.8 GiB)",
            url: "https://huggingface.co/ggml-org/models/resolve/main/phi-2/ggml-model-q8_0.gguf?download=true",
            filename: "phi-2-q8_0.gguf", status: "download"
        ),

        Model(
            name: "Mistral-7B-v0.1 (Q4_0, 3.8 GiB)",
            url: "https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q4_0.gguf?download=true",
            filename: "mistral-7b-v0.1.Q4_0.gguf", status: "download"
        ),
        Model(
            name: "OpenHermes-2.5-Mistral-7B (Q3_K_M, 3.52 GiB)",
            url: "https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF/resolve/main/openhermes-2.5-mistral-7b.Q3_K_M.gguf?download=true",
            filename: "openhermes-2.5-mistral-7b.Q3_K_M.gguf", status: "download"
        )
    ]
    func loadModel(modelUrl: URL?) async -> Bool {
        guard let modelUrl else {
            messageLog += "Load a model from the list below\n"
            return false
        }

        guard !isLoadingModel else {
            messageLog += "A model is already loading. Please wait.\n"
            return false
        }

        isLoadingModel = true
        loadingStatus = "Loading \(modelUrl.lastPathComponent)..."
        messageLog += "\nLoading \(modelUrl.lastPathComponent)...\n"
        llamaContext = nil
        loadedModelName = nil

        let path = modelUrl.path()
        var didLoad = false
        do {
            let context = try await Task.detached(priority: .userInitiated) {
                try LlamaContext.create_context(path: path)
            }.value

            llamaContext = context
            loadedModelName = modelUrl.lastPathComponent
            loadingStatus = "Loaded \(modelUrl.lastPathComponent)"
            messageLog += "Loaded model \(modelUrl.lastPathComponent)\n"
            recordSelfTest("[SelfTest] Loaded model \(modelUrl.lastPathComponent)")
            print("[SelfTest] Loaded model \(modelUrl.lastPathComponent)")
            updateDownloadedModels(modelName: modelUrl.lastPathComponent, status: "downloaded")
            didLoad = true
        } catch {
            loadingStatus = "Failed to load \(modelUrl.lastPathComponent)"
            messageLog += "Load failed: \(error.localizedDescription)\n"
            recordSelfTest("[SelfTest] Load failed: \(error.localizedDescription)")
        }

        isLoadingModel = false
        return didLoad
    }

    func importModel(from sourceURL: URL) async {
        guard !isLoadingModel else {
            messageLog += "A model is already loading. Please wait.\n"
            return
        }

        let destinationURL = getDocumentsDirectory().appendingPathComponent(sourceURL.lastPathComponent)
        let sourcePath = sourceURL.path
        let destinationPath = destinationURL.path

        isLoadingModel = true
        loadingStatus = "Copying \(sourceURL.lastPathComponent)..."
        messageLog += "\nCopying \(sourceURL.lastPathComponent) into app storage...\n"

        do {
            try await Task.detached(priority: .userInitiated) {
                if FileManager.default.fileExists(atPath: destinationPath) {
                    try FileManager.default.removeItem(atPath: destinationPath)
                }
                try FileManager.default.copyItem(atPath: sourcePath, toPath: destinationPath)
            }.value

            let modelName = destinationURL.deletingPathExtension().lastPathComponent
            downloadedModels.removeAll { $0.filename == destinationURL.lastPathComponent }
            downloadedModels.append(Model(name: modelName, url: "", filename: destinationURL.lastPathComponent, status: "downloaded"))
            loadingStatus = "Copied \(destinationURL.lastPathComponent)"
            messageLog += "Copied model into app storage.\n"
        } catch {
            loadingStatus = "Failed to copy \(sourceURL.lastPathComponent)"
            messageLog += "Import failed: \(error.localizedDescription)\n"
            isLoadingModel = false
            return
        }

        isLoadingModel = false
        _ = await loadModel(modelUrl: destinationURL)
    }


    private func updateDownloadedModels(modelName: String, status: String) {
        undownloadedModels.removeAll { $0.name == modelName }
    }


    func complete(text: String) async {
        if isLoadingModel {
            messageLog += "Model is still loading. Please wait.\n"
            return
        }

        if isGenerating {
            messageLog += "Generation is still running. Please wait.\n"
            return
        }

        let prompt = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else {
            messageLog += "Enter a prompt before sending.\n"
            return
        }

        guard let llamaContext else {
            messageLog += "Load a model before sending a prompt.\n"
            return
        }

        stopRequested = false
        isGenerating = true
        defer {
            isGenerating = false
            stopRequested = false
        }

        let t_start = DispatchTime.now().uptimeNanoseconds
        let didStart = await llamaContext.completion_init(text: prompt)
        let t_heat_end = DispatchTime.now().uptimeNanoseconds
        let t_heat = Double(t_heat_end - t_start) / NS_PER_S

        guard didStart else {
            messageLog += "Could not start generation.\n"
            await llamaContext.clear()
            return
        }

        messageLog += "\(prompt)"

        while await !llamaContext.is_done {
            if stopRequested {
                await llamaContext.stop_completion()
                break
            }
            let result = await llamaContext.completion_loop()
            messageLog += "\(result)"
        }

        let t_end = DispatchTime.now().uptimeNanoseconds
        let t_generation = Double(t_end - t_heat_end) / NS_PER_S
        let decoded = max(1, await llamaContext.n_decode)
        let tokens_per_second = Double(decoded) / max(t_generation, 0.001)

        await llamaContext.clear()

        if stopRequested {
            messageLog += "\n\nStopped\n"
        } else {
            messageLog += """
                \n
                Done
                Heat up took \(t_heat)s
                Generated \(tokens_per_second) t/s\n
                """
        }
    }

    func stopGeneration() {
        guard isGenerating else {
            return
        }

        stopRequested = true
        messageLog += "\nStopping...\n"
        Task {
            await llamaContext?.stop_completion()
        }
    }

    private func runQuickCompletionSelfTest(prompt: String, maxTokens: Int) async {
        guard let llamaContext else {
            messageLog += "[SelfTest] completion skipped: no model loaded\n"
            return
        }

        messageLog += "[SelfTest] completion prompt: \(prompt)\n"
        let didStart = await llamaContext.completion_init(text: prompt)
        guard didStart else {
            messageLog += "[SelfTest] completion failed to start\n"
            recordSelfTest("[SelfTest] completion failed")
            return
        }
        messageLog += prompt

        for _ in 0..<maxTokens {
            if await llamaContext.is_done {
                break
            }
            let result = await llamaContext.completion_loop()
            messageLog += result
        }

        await llamaContext.clear()
        messageLog += "\n[SelfTest] completion ok\n"
        recordSelfTest("[SelfTest] completion ok")
        print("[SelfTest] completion ok")
    }

    private func runQuickBenchmarkSelfTest(pp: Int, tg: Int) async {
        guard let llamaContext else {
            messageLog += "[SelfTest] bench skipped: no model loaded\n"
            return
        }

        messageLog += "[SelfTest] bench start\n"
        let result = await llamaContext.bench(pp: pp, tg: tg, pl: 1)
        messageLog += result
        messageLog += "\n[SelfTest] bench ok\n"
        recordSelfTest("[SelfTest] bench ok")
        print("[SelfTest] bench ok")
    }

    func bench() async {
        if isLoadingModel {
            messageLog += "Model is still loading. Please wait.\n"
            return
        }

        if isGenerating {
            messageLog += "Generation is still running. Please wait.\n"
            return
        }

        guard let llamaContext else {
            messageLog += "Load a model before benchmarking.\n"
            return
        }

        isGenerating = true
        defer {
            isGenerating = false
        }

        messageLog += "\n"
        messageLog += "Running benchmark...\n"
        messageLog += "Model info: "
        messageLog += await llamaContext.model_info() + "\n"

        let t_start = DispatchTime.now().uptimeNanoseconds
        let _ = await llamaContext.bench(pp: 8, tg: 4, pl: 1) // heat up
        let t_end = DispatchTime.now().uptimeNanoseconds

        let t_heat = Double(t_end - t_start) / NS_PER_S
        messageLog += "Heat up time: \(t_heat) seconds, please wait...\n"

        // if more than 5 seconds, then we're probably running on a slow device
        if t_heat > 5.0 {
            messageLog += "Heat up time is too long, aborting benchmark\n"
            return
        }

        let result = await llamaContext.bench(pp: 512, tg: 128, pl: 1, nr: 3)

        messageLog += "\(result)"
        messageLog += "\n"
    }

    func clear() async {
        guard let llamaContext else {
            return
        }

        await llamaContext.clear()
        messageLog = ""
    }
}
