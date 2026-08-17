import SwiftUI

private let oscarGold = Color(red: 0.04, green: 0.48, blue: 0.90)
private let oscarPanel = Color(red: 0.95, green: 0.97, blue: 1.0)
private let chatBackground = Color(red: 0.985, green: 0.988, blue: 0.992)
private let chatText = Color(red: 0.07, green: 0.09, blue: 0.12)
private let chatMuted = Color(red: 0.45, green: 0.49, blue: 0.56)

struct ContentView: View {
    @StateObject var llamaState = LlamaState()
    @State private var multiLineText = ""
    @FocusState private var isPromptFocused: Bool

    var body: some View {
        NavigationView {
            ZStack {
                chatBackground
                .ignoresSafeArea()

                VStack(spacing: 0) {
                    chatHeader
                    chatContent
                    composerBar
                }
            }
            .contentShape(Rectangle())
            .onTapGesture {
                dismissKeyboard()
            }
            .navigationBarHidden(true)
        }
    }

    private var chatHeader: some View {
        HStack(spacing: 12) {
            NavigationLink(destination: DrawerView(llamaState: llamaState)) {
                Image(systemName: "line.3.horizontal")
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundStyle(chatText)
                    .frame(width: 46, height: 46)
                    .background(Color.white)
                    .clipShape(Circle())
                    .shadow(color: Color.black.opacity(0.06), radius: 12, x: 0, y: 6)
            }

            HStack(spacing: 6) {
                Text(llamaState.isGenerating ? "Thinking" : "OSCAR")
                    .font(.subheadline.weight(.semibold))
                if llamaState.isLoadingModel {
                    ProgressView()
                        .scaleEffect(0.7)
                }
            }
            .foregroundStyle(oscarGold)
            .padding(.horizontal, 16)
            .frame(height: 44)
            .background(Color.white)
            .clipShape(Capsule())
            .shadow(color: Color.black.opacity(0.06), radius: 12, x: 0, y: 6)

            Spacer()

            statusBubble
        }
        .padding(.horizontal, 18)
        .padding(.top, 12)
        .padding(.bottom, 8)
    }

    private var statusBubble: some View {
        Image(systemName: llamaState.loadedModelName == nil ? "circle.dashed" : "checkmark.circle")
            .font(.system(size: 22, weight: .semibold))
            .foregroundStyle(llamaState.loadedModelName == nil ? chatText : oscarGold)
            .frame(width: 46, height: 46)
            .background(Color.white)
            .clipShape(Circle())
            .shadow(color: Color.black.opacity(0.06), radius: 12, x: 0, y: 6)
    }

    private var chatContent: some View {
        ScrollView(.vertical, showsIndicators: true) {
            VStack(spacing: 18) {
                if llamaState.messageLog.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ||
                    llamaState.messageLog == "OSCAR runtime ready. Import or download a GGUF model to begin.\n" {
                    emptyState
                        .frame(maxWidth: .infinity)
                        .padding(.top, 140)
                } else {
                    VStack(alignment: .leading, spacing: 10) {
                        Text(llamaState.loadedModelName ?? "OSCAR")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(chatMuted)
                        Text(llamaState.messageLog)
                            .font(.system(size: 14, design: .monospaced))
                            .foregroundStyle(chatText)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .padding(16)
                    .background(Color.white)
                    .clipShape(RoundedRectangle(cornerRadius: 22, style: .continuous))
                    .shadow(color: Color.black.opacity(0.04), radius: 18, x: 0, y: 8)
                    .padding(.horizontal, 18)
                    .padding(.top, 18)
                }
            }
        }
        .scrollDismissesKeyboard(.interactively)
        .onTapGesture {
            dismissKeyboard()
        }
    }

    private var emptyState: some View {
        VStack(spacing: 14) {
            Image("oscar_logo_kv_transparent")
                .resizable()
                .scaledToFit()
                .frame(width: 56, height: 56)
                .clipShape(RoundedRectangle(cornerRadius: 15, style: .continuous))
                .shadow(color: oscarGold.opacity(0.18), radius: 18, x: 0, y: 8)

            Text(llamaState.loadedModelName == nil ? "Load a GGUF model to begin" : "Ask OSCAR anything")
                .font(.headline.weight(.semibold))
                .foregroundStyle(chatText)

            Text(llamaState.loadedModelName ?? llamaState.loadingStatus)
                .font(.subheadline)
                .multilineTextAlignment(.center)
                .foregroundStyle(chatMuted)
                .padding(.horizontal, 36)
        }
    }

    private var composerBar: some View {
        VStack(spacing: 10) {
            HStack(spacing: 10) {
                Button(action: bench) {
                    Label("Bench", systemImage: "speedometer")
                }
                .disabled(llamaState.isLoadingModel || llamaState.isGenerating || llamaState.loadedModelName == nil)
                .buttonStyle(ChatUtilityButtonStyle())

                Button(action: clear) {
                    Label("Clear", systemImage: "trash")
                }
                .disabled(llamaState.isLoadingModel || llamaState.isGenerating)
                .buttonStyle(ChatUtilityButtonStyle())

                Button("Copy") {
                    UIPasteboard.general.string = llamaState.messageLog
                }
                .buttonStyle(ChatUtilityButtonStyle())
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            HStack(alignment: .center, spacing: 10) {
                NavigationLink(destination: DrawerView(llamaState: llamaState)) {
                    Image(systemName: "plus")
                        .font(.system(size: 22, weight: .medium))
                        .foregroundStyle(chatText)
                        .frame(width: 38, height: 38)
                        .background(Color.white)
                        .clipShape(Circle())
                }

                ZStack(alignment: .leading) {
                    if multiLineText.isEmpty {
                        Text("Ask OSCAR")
                            .foregroundStyle(chatMuted.opacity(0.75))
                            .padding(.horizontal, 14)
                    }

                    TextEditor(text: $multiLineText)
                        .frame(height: 38)
                        .focused($isPromptFocused)
                        .scrollContentBackground(.hidden)
                        .foregroundStyle(chatText)
                        .padding(.horizontal, 9)
                }
                .background(Color.white)
                .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 20, style: .continuous)
                        .stroke(Color.black.opacity(0.06), lineWidth: 1)
                )
                .shadow(color: Color.black.opacity(0.05), radius: 10, x: 0, y: 4)

                Button(action: sendText) {
                    Image(systemName: llamaState.isGenerating ? "stop.fill" : "arrow.up")
                        .font(.system(size: 16, weight: .bold))
                        .foregroundStyle(Color.white)
                        .frame(width: 38, height: 38)
                        .background(sendButtonColor)
                        .clipShape(Circle())
                }
                .disabled(sendDisabled)
            }
        }
        .padding(.horizontal, 14)
        .padding(.top, 10)
        .padding(.bottom, 10)
        .background(.ultraThinMaterial)
    }

    private var sendDisabled: Bool {
        if llamaState.isGenerating {
            return false
        }

        return llamaState.isLoadingModel ||
            llamaState.loadedModelName == nil ||
            multiLineText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var sendButtonColor: Color {
        sendDisabled ? Color.black.opacity(0.25) : Color.black
    }

    func sendText() {
        if llamaState.isGenerating {
            llamaState.stopGeneration()
            return
        }

        dismissKeyboard()
        Task {
            await llamaState.complete(text: multiLineText)
            multiLineText = ""
        }
    }

    func bench() {
        Task {
            await llamaState.bench()
        }
    }

    func clear() {
        dismissKeyboard()
        Task {
            await llamaState.clear()
        }
    }

    func dismissKeyboard() {
        isPromptFocused = false
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
    }

    struct DrawerView: View {
        @ObservedObject var llamaState: LlamaState
        @State private var showingHelp = false

        func delete(at offsets: IndexSet) {
            offsets.forEach { offset in
                let model = llamaState.downloadedModels[offset]
                let fileURL = getDocumentsDirectory().appendingPathComponent(model.filename)
                do {
                    try FileManager.default.removeItem(at: fileURL)
                } catch {
                    print("Error deleting file: \(error)")
                }
            }

            llamaState.downloadedModels.remove(atOffsets: offsets)
        }

        func getDocumentsDirectory() -> URL {
            let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
            return paths[0]
        }

        var body: some View {
            List {
                Section {
                    VStack(alignment: .leading, spacing: 12) {
                        Image("oscar_logo_kv_transparent")
                            .resizable()
                            .scaledToFit()
                            .frame(maxWidth: .infinity)
                            .frame(height: 130)

                        Text("Model Storage")
                            .font(.headline)

                        Text("The app installs the OSCAR runtime only. GGUF model files are large, so they live in this app's Documents folder. Import an existing GGUF from Files or download one here.")
                            .font(.footnote)
                            .foregroundStyle(.secondary)

                        HStack(spacing: 10) {
                            if llamaState.isLoadingModel {
                                ProgressView()
                            } else {
                                Image(systemName: "bolt.horizontal.circle.fill")
                                    .foregroundStyle(oscarGold)
                            }

                            VStack(alignment: .leading, spacing: 2) {
                                Text(llamaState.loadingStatus)
                                    .font(.footnote)
                                    .fontWeight(.medium)
                                Text("Large GGUF files can take tens of seconds or longer to map, upload, and initialize on iPhone.")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                        .padding(10)
                        .background(Color(.secondarySystemGroupedBackground))
                        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                    }
                    .padding(.vertical, 8)
                }

                Section(header: Text("Import Local Model")) {
                    LoadCustomButton(llamaState: llamaState)
                }

                Section(header: Text("Download From Hugging Face")) {
                    VStack(alignment: .leading, spacing: 8) {
                        InputButton(llamaState: llamaState)
                        Text("Paste a direct .gguf download URL if you already have one.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                Section(header: Text("On This iPhone")) {
                    ForEach(llamaState.downloadedModels) { model in
                        DownloadButton(llamaState: llamaState, modelName: model.name, modelUrl: model.url, filename: model.filename)
                    }
                    .onDelete(perform: delete)
                }

                Section(header: Text("Recommended Downloads")) {
                    ForEach(llamaState.undownloadedModels) { model in
                        DownloadButton(llamaState: llamaState, modelName: model.name, modelUrl: model.url, filename: model.filename)
                    }
                }
            }
            .listStyle(GroupedListStyle())
            .navigationBarTitle("Models", displayMode: .inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Help") {
                        showingHelp = true
                    }
                }
            }
            .sheet(isPresented: $showingHelp) {
                NavigationView {
                    VStack(alignment: .leading) {
                        VStack(alignment: .leading, spacing: 14) {
                            Text("1. Use GGUF model files only.")
                            Text("2. OSCAR INT2 models should include rot-kv in the filename so the app enables q2_0 KV automatically.")
                            Text("3. Large models are not bundled with the app. Import them from Files or download them into app storage.")
                        }
                        .padding()
                        Spacer()
                    }
                    .navigationTitle("Help")
                    .navigationBarTitleDisplayMode(.inline)
                    .toolbar {
                        ToolbarItem(placement: .navigationBarTrailing) {
                            Button("Done") {
                                showingHelp = false
                            }
                        }
                    }
                }
            }
        }
    }
}

private struct OscarActionButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled

    let isPrimary: Bool

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.footnote.weight(.semibold))
            .padding(.vertical, 10)
            .foregroundStyle(isEnabled ? (isPrimary ? Color.black : oscarGold) : Color.white.opacity(0.35))
            .background(isEnabled ? (isPrimary ? oscarGold : Color.white.opacity(0.08)) : Color.white.opacity(0.05))
            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .stroke(isEnabled ? oscarGold.opacity(isPrimary ? 0 : 0.45) : Color.white.opacity(0.12), lineWidth: 1)
            )
            .opacity(configuration.isPressed ? 0.72 : 1)
    }
}

private struct ChatUtilityButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.subheadline.weight(.semibold))
            .foregroundStyle(isEnabled ? chatText : chatMuted.opacity(0.45))
            .padding(.horizontal, 14)
            .frame(height: 36)
            .background(isEnabled ? Color.white : Color.white.opacity(0.55))
            .clipShape(Capsule())
            .overlay(
                Capsule()
                    .stroke(Color.black.opacity(0.06), lineWidth: 1)
            )
            .shadow(color: Color.black.opacity(0.04), radius: 8, x: 0, y: 3)
            .opacity(configuration.isPressed ? 0.72 : 1)
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
