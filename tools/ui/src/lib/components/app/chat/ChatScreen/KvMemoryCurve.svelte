<script lang="ts">
	interface Props {
		class?: string;
	}

	let { class: className = '' }: Props = $props();

	const layers = 48;
	const kvWidth = 512;
	const contextTokens = 262144;
	const highPrecisionTokens = 64 + 256;
	const bf16BytesPerValue = 2;
	const int2BytesPerValue = 0.25;

	const bf16BytesPerToken = layers * kvWidth * 2 * bf16BytesPerValue;
	const int2BytesPerToken = layers * kvWidth * 2 * int2BytesPerValue;

	function toGib(bytes: number): number {
		return bytes / 1024 ** 3;
	}

	function bf16Gib(tokens: number): number {
		return toGib(tokens * bf16BytesPerToken);
	}

	function oscarGib(tokens: number): number {
		const hpTokens = Math.min(tokens, highPrecisionTokens);
		const int2Tokens = Math.max(0, tokens - highPrecisionTokens);

		return toGib(hpTokens * bf16BytesPerToken + int2Tokens * int2BytesPerToken);
	}

	const bf16AtMax = bf16Gib(contextTokens);
	const oscarAtMax = oscarGib(contextTokens);
	const savedGib = bf16AtMax - oscarAtMax;
	const compression = bf16AtMax / oscarAtMax;
	const oscarWidthPercent = (oscarAtMax / bf16AtMax) * 100;
</script>

<div
	class="mx-auto mt-5 max-w-md rounded-2xl border border-slate-200/80 bg-white/75 p-4 text-left shadow-sm backdrop-blur-sm {className}"
>
	<div class="flex items-start justify-between gap-3">
		<div>
			<p class="text-sm font-medium text-foreground">At 256K context</p>
			<p class="mt-1 text-xs text-muted-foreground">
				Estimated KV memory for Gemma 12B
			</p>
		</div>

		<div class="rounded-full bg-cyan-50 px-2.5 py-1 text-xs font-medium text-cyan-700">
			~{compression.toFixed(1)}x smaller
		</div>
	</div>

	<div class="mt-4 space-y-3">
		<div>
			<div class="mb-1 flex items-center justify-between text-xs">
				<span class="text-muted-foreground">BF16 KV</span>
				<span class="font-medium text-slate-600">{bf16AtMax.toFixed(1)} GiB</span>
			</div>
			<div class="h-2.5 rounded-full bg-slate-200">
				<div class="h-2.5 rounded-full bg-slate-400" style="width: 100%"></div>
			</div>
		</div>

		<div>
			<div class="mb-1 flex items-center justify-between text-xs">
				<span class="text-cyan-700">OSCAR 2-bit KV</span>
				<span class="font-medium text-cyan-700">{oscarAtMax.toFixed(1)} GiB</span>
			</div>
			<div class="h-2.5 rounded-full bg-cyan-50">
				<div class="h-2.5 rounded-full bg-cyan-600" style={`width: ${oscarWidthPercent}%`}></div>
			</div>
		</div>
	</div>

	<div class="mt-4 rounded-xl bg-cyan-50 px-3 py-2 text-center">
		<span class="text-xs text-cyan-700">Saved at 256K</span>
		<div class="text-xl font-semibold tracking-tight text-cyan-900">~{savedGib.toFixed(1)} GiB</div>
	</div>
</div>
