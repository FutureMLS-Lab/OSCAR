import Darwin
import Foundation
import llama

enum LlamaError: Error {
    case couldNotInitializeContext
}

func llama_batch_clear(_ batch: inout llama_batch) {
    batch.n_tokens = 0
}

func llama_batch_add(_ batch: inout llama_batch, _ id: llama_token, _ pos: llama_pos, _ seq_ids: [llama_seq_id], _ logits: Bool) {
    batch.token   [Int(batch.n_tokens)] = id
    batch.pos     [Int(batch.n_tokens)] = pos
    batch.n_seq_id[Int(batch.n_tokens)] = Int32(seq_ids.count)
    for i in 0..<seq_ids.count {
        batch.seq_id[Int(batch.n_tokens)]![Int(i)] = seq_ids[i]
    }
    batch.logits  [Int(batch.n_tokens)] = logits ? 1 : 0

    batch.n_tokens += 1
}

actor LlamaContext {
    private var model: OpaquePointer
    private var context: OpaquePointer
    private var vocab: OpaquePointer
    private var sampling: UnsafeMutablePointer<llama_sampler>
    private var batch: llama_batch
    private var tokens_list: [llama_token]
    var is_done: Bool = false

    /// This variable is used to store temporarily invalid cchars
    private var temporary_invalid_cchars: [CChar]

    /// Budget for newly generated tokens. Thinking models routinely spend ~1k tokens
    /// inside <think> before answering, so this needs headroom.
    var n_max_new: Int32 = 4096
    private var n_len: Int32 = 0
    var n_cur: Int32 = 0

    var n_decode: Int32 = 0

    private static func shouldUseOscarInt2KV(path: String) -> Bool {
        let lowercasedPath = path.lowercased()
        return lowercasedPath.contains("rot-kv") || lowercasedPath.contains("oscar")
    }

    private static func configureOscarInt2KVEnvironment() {
        setenv("LLAMA_KV_FUSED_FA", "1", 1)
        setenv("LLAMA_KV_NO_HADAMARD", "1", 1)
        setenv("LLAMA_KV_CLIP_RATIO", "0.96", 1)
        setenv("LLAMA_KV_HP_SINK", "64", 1)
        setenv("LLAMA_KV_HP_RECENT", "256", 1)
    }

    private let batch_capacity = 512

    init(model: OpaquePointer, context: OpaquePointer) {
        self.model = model
        self.context = context
        self.tokens_list = []
        self.batch = llama_batch_init(Int32(batch_capacity), 0, 1)
        self.temporary_invalid_cchars = []
        let sparams = llama_sampler_chain_default_params()
        self.sampling = llama_sampler_chain_init(sparams)
        // Qwen3 / Qwen3-Thinking recommended sampling. Without the top-k/top-p
        // truncation the 151k-entry tail stays in play and derails long generations.
        llama_sampler_chain_add(self.sampling, llama_sampler_init_top_k(20))
        llama_sampler_chain_add(self.sampling, llama_sampler_init_top_p(0.95, 1))
        llama_sampler_chain_add(self.sampling, llama_sampler_init_temp(0.6))
        llama_sampler_chain_add(self.sampling, llama_sampler_init_dist(1234))
        vocab = llama_model_get_vocab(model)
    }

    deinit {
        llama_sampler_free(sampling)
        llama_batch_free(batch)
        llama_model_free(model)
        llama_free(context)
        llama_backend_free()
    }

    static func create_context(path: String) throws -> LlamaContext {
        let useOscarInt2KV = shouldUseOscarInt2KV(path: path)
        if useOscarInt2KV {
            configureOscarInt2KVEnvironment()
            print("OSCAR INT2 KV enabled for \(path)")
        }

        llama_backend_init()
        var model_params = llama_model_default_params()

#if targetEnvironment(simulator)
        model_params.n_gpu_layers = 0
        print("Running on simulator, force use n_gpu_layers = 0")
#endif
        let model = llama_model_load_from_file(path, model_params)
        guard let model else {
            print("Could not load model at \(path)")
            throw LlamaError.couldNotInitializeContext
        }

        let n_threads = max(1, min(8, ProcessInfo.processInfo.processorCount - 2))
        print("Using \(n_threads) threads")

        var ctx_params = llama_context_default_params()
        ctx_params.n_ctx = useOscarInt2KV ? 8192 : 4096
        ctx_params.n_batch = 256
        ctx_params.n_ubatch = 128
        ctx_params.n_threads       = Int32(n_threads)
        ctx_params.n_threads_batch = Int32(n_threads)
        print("Context params: n_ctx=\(ctx_params.n_ctx), n_batch=\(ctx_params.n_batch), n_ubatch=\(ctx_params.n_ubatch)")

        if useOscarInt2KV {
            ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED
            ctx_params.type_k = GGML_TYPE_Q2_0
            ctx_params.type_v = GGML_TYPE_Q2_0
        }

        let context = llama_init_from_model(model, ctx_params)
        guard let context else {
            print("Could not load context for \(path)")
            throw LlamaError.couldNotInitializeContext
        }

        return LlamaContext(model: model, context: context)
    }

    func model_info() -> String {
        let result = UnsafeMutablePointer<Int8>.allocate(capacity: 256)
        result.initialize(repeating: Int8(0), count: 256)
        defer {
            result.deallocate()
        }

        // TODO: this is probably very stupid way to get the string from C

        let nChars = llama_model_desc(model, result, 256)
        let bufferPointer = UnsafeBufferPointer(start: result, count: Int(nChars))

        var SwiftString = ""
        for char in bufferPointer {
            SwiftString.append(Character(UnicodeScalar(UInt8(char))))
        }

        return SwiftString
    }

    func get_n_tokens() -> Int32 {
        return batch.n_tokens;
    }

    /// Wrap the user turn in the model's own chat template. Returns nil when the model
    /// ships no template, or when it is not one of the formats llama.cpp knows.
    private func format_as_chat(_ text: String) -> String? {
        guard let tmpl = llama_model_chat_template(model, nil) else {
            return nil
        }

        return text.withCString { (content: UnsafePointer<CChar>) -> String? in
            return "user".withCString { (role: UnsafePointer<CChar>) -> String? in
                var message = llama_chat_message(role: role, content: content)
                var buffer = [CChar](repeating: 0, count: max(2048, text.utf8.count * 4))
                var n = llama_chat_apply_template(tmpl, &message, 1, true, &buffer, Int32(buffer.count))

                if n > Int32(buffer.count) {
                    buffer = [CChar](repeating: 0, count: Int(n) + 1)
                    n = llama_chat_apply_template(tmpl, &message, 1, true, &buffer, Int32(buffer.count))
                }

                guard n > 0 else {
                    return nil
                }

                return String(cString: Array(buffer[0..<Int(n)]) + [0])
            }
        }
    }

    func completion_init(text: String) -> Bool {
        print("attempting to complete \"\(text)\"")

        is_done = false
        n_cur = 0
        n_decode = 0
        temporary_invalid_cchars = []

        let formatted = format_as_chat(text)
        if formatted == nil {
            print("no usable chat template, falling back to raw text completion")
        }
        tokens_list = tokenize(text: formatted ?? text, add_bos: true, parse_special: formatted != nil)

        guard !tokens_list.isEmpty else {
            print("tokenize() returned no tokens")
            is_done = true
            return false
        }

        let n_ctx = Int32(llama_n_ctx(context))
        let n_prompt = Int32(tokens_list.count)

        guard n_prompt < n_ctx else {
            print("error: prompt is \(n_prompt) tokens, does not fit in a context of \(n_ctx)")
            is_done = true
            return false
        }

        n_len = min(n_ctx, n_prompt + n_max_new)
        print("\n n_prompt = \(n_prompt), n_len = \(n_len), n_ctx = \(n_ctx)")

        llama_batch_clear(&batch)
        llama_memory_clear(llama_get_memory(context), true)

        // llama_decode() rejects batches larger than n_batch, so feed the prompt in chunks
        // and only ask for logits on the very last token.
        let chunk_size = min(Int(llama_n_batch(context)), batch_capacity)
        var i = 0
        while i < tokens_list.count {
            let n_chunk = min(chunk_size, tokens_list.count - i)

            llama_batch_clear(&batch)
            for j in 0..<n_chunk {
                llama_batch_add(&batch, tokens_list[i + j], Int32(i + j), [0], false)
            }

            i += n_chunk
            if i == tokens_list.count {
                batch.logits[Int(batch.n_tokens) - 1] = 1 // true
            }

            if llama_decode(context, batch) != 0 {
                print("llama_decode() failed")
                is_done = true
                return false
            }
        }

        n_cur = n_prompt
        return true
    }

    func completion_loop() -> String {
        var new_token_id: llama_token = 0

        guard !is_done, batch.n_tokens > 0 else {
            is_done = true
            return ""
        }

        new_token_id = llama_sampler_sample(sampling, context, batch.n_tokens - 1)

        if llama_vocab_is_eog(vocab, new_token_id) || n_cur == n_len {
            print("\n")
            is_done = true
            let new_token_str = String(cString: temporary_invalid_cchars + [0])
            temporary_invalid_cchars.removeAll()
            return new_token_str
        }

        let new_token_cchars = token_to_piece(token: new_token_id)
        temporary_invalid_cchars.append(contentsOf: new_token_cchars)
        let new_token_str: String
        if let string = String(validatingUTF8: temporary_invalid_cchars + [0]) {
            temporary_invalid_cchars.removeAll()
            new_token_str = string
        } else if (0 ..< temporary_invalid_cchars.count).contains(where: {$0 != 0 && String(validatingUTF8: Array(temporary_invalid_cchars.suffix($0)) + [0]) != nil}) {
            // in this case, at least the suffix of the temporary_invalid_cchars can be interpreted as UTF8 string
            let string = String(cString: temporary_invalid_cchars + [0])
            temporary_invalid_cchars.removeAll()
            new_token_str = string
        } else {
            new_token_str = ""
        }
        print(new_token_str)
        // tokens_list.append(new_token_id)

        llama_batch_clear(&batch)
        llama_batch_add(&batch, new_token_id, n_cur, [0], true)

        n_decode += 1
        n_cur    += 1

        if llama_decode(context, batch) != 0 {
            print("failed to evaluate llama!")
            is_done = true
        }

        return new_token_str
    }

    func stop_completion() {
        is_done = true
    }

    func bench(pp: Int, tg: Int, pl: Int, nr: Int = 1) -> String {
        var pp_avg: Double = 0
        var tg_avg: Double = 0

        var pp_std: Double = 0
        var tg_std: Double = 0

        for _ in 0..<nr {
            // bench prompt processing

            llama_batch_clear(&batch)

            let n_tokens = pp

            for i in 0..<n_tokens {
                llama_batch_add(&batch, 0, Int32(i), [0], false)
            }
            batch.logits[Int(batch.n_tokens) - 1] = 1 // true

            llama_memory_clear(llama_get_memory(context), false)

            let t_pp_start = DispatchTime.now().uptimeNanoseconds / 1000;

            if llama_decode(context, batch) != 0 {
                print("llama_decode() failed during prompt")
            }
            llama_synchronize(context)

            let t_pp_end = DispatchTime.now().uptimeNanoseconds / 1000;

            // bench text generation

            llama_memory_clear(llama_get_memory(context), false)

            let t_tg_start = DispatchTime.now().uptimeNanoseconds / 1000;

            for i in 0..<tg {
                llama_batch_clear(&batch)

                for j in 0..<pl {
                    llama_batch_add(&batch, 0, Int32(i), [Int32(j)], true)
                }

                if llama_decode(context, batch) != 0 {
                    print("llama_decode() failed during text generation")
                }
                llama_synchronize(context)
            }

            let t_tg_end = DispatchTime.now().uptimeNanoseconds / 1000;

            llama_memory_clear(llama_get_memory(context), false)

            let t_pp = Double(t_pp_end - t_pp_start) / 1000000.0
            let t_tg = Double(t_tg_end - t_tg_start) / 1000000.0

            let speed_pp = Double(pp)    / t_pp
            let speed_tg = Double(pl*tg) / t_tg

            pp_avg += speed_pp
            tg_avg += speed_tg

            pp_std += speed_pp * speed_pp
            tg_std += speed_tg * speed_tg

            print("pp \(speed_pp) t/s, tg \(speed_tg) t/s")
        }

        pp_avg /= Double(nr)
        tg_avg /= Double(nr)

        if nr > 1 {
            pp_std = sqrt(pp_std / Double(nr - 1) - pp_avg * pp_avg * Double(nr) / Double(nr - 1))
            tg_std = sqrt(tg_std / Double(nr - 1) - tg_avg * tg_avg * Double(nr) / Double(nr - 1))
        } else {
            pp_std = 0
            tg_std = 0
        }

        let model_desc     = model_info();
        let model_size     = String(format: "%.2f GiB", Double(llama_model_size(model)) / 1024.0 / 1024.0 / 1024.0);
        let model_n_params = String(format: "%.2f B", Double(llama_model_n_params(model)) / 1e9);
        let backend        = "Metal";
        let pp_avg_str     = String(format: "%.2f", pp_avg);
        let tg_avg_str     = String(format: "%.2f", tg_avg);
        let pp_std_str     = String(format: "%.2f", pp_std);
        let tg_std_str     = String(format: "%.2f", tg_std);

        var result = ""

        result += String("| model | size | params | backend | test | t/s |\n")
        result += String("| --- | --- | --- | --- | --- | --- |\n")
        result += String("| \(model_desc) | \(model_size) | \(model_n_params) | \(backend) | pp \(pp) | \(pp_avg_str) ± \(pp_std_str) |\n")
        result += String("| \(model_desc) | \(model_size) | \(model_n_params) | \(backend) | tg \(tg) | \(tg_avg_str) ± \(tg_std_str) |\n")

        return result;
    }

    func clear() {
        tokens_list.removeAll()
        temporary_invalid_cchars.removeAll()
        llama_memory_clear(llama_get_memory(context), true)
    }

    private func tokenize(text: String, add_bos: Bool, parse_special: Bool = false) -> [llama_token] {
        let utf8Count = text.utf8.count
        let n_tokens = utf8Count + (add_bos ? 1 : 0) + 1
        let tokens = UnsafeMutablePointer<llama_token>.allocate(capacity: n_tokens)
        let tokenCount = llama_tokenize(vocab, text, Int32(utf8Count), tokens, Int32(n_tokens), add_bos, parse_special)

        var swiftTokens: [llama_token] = []
        for i in 0..<tokenCount {
            swiftTokens.append(tokens[Int(i)])
        }

        tokens.deallocate()

        return swiftTokens
    }

    /// - note: The result does not contain null-terminator
    private func token_to_piece(token: llama_token) -> [CChar] {
        let result = UnsafeMutablePointer<Int8>.allocate(capacity: 8)
        result.initialize(repeating: Int8(0), count: 8)
        defer {
            result.deallocate()
        }
        let nTokens = llama_token_to_piece(vocab, token, result, 8, 0, false)

        if nTokens < 0 {
            let newResult = UnsafeMutablePointer<Int8>.allocate(capacity: Int(-nTokens))
            newResult.initialize(repeating: Int8(0), count: Int(-nTokens))
            defer {
                newResult.deallocate()
            }
            let nNewTokens = llama_token_to_piece(vocab, token, newResult, -nTokens, 0, false)
            let bufferPointer = UnsafeBufferPointer(start: newResult, count: Int(nNewTokens))
            return Array(bufferPointer)
        } else {
            let bufferPointer = UnsafeBufferPointer(start: result, count: Int(nTokens))
            return Array(bufferPointer)
        }
    }
}
