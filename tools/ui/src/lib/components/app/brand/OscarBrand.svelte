<script lang="ts">
	import { base } from '$app/paths';

	interface Props {
		class?: string;
		logoClass?: string;
		showTagline?: boolean;
	}

	let {
		class: className = '',
		logoClass = 'h-12 w-12',
		showTagline = true
	}: Props = $props();
</script>

<div class="flex min-w-0 items-center gap-3 {className}">
	<div
		class="relative flex shrink-0 items-center justify-center rounded-2xl border border-cyan-100 bg-white/80 p-1.5 shadow-sm {logoClass}"
	>
		<img
			class="relative h-full w-full object-contain"
			src="{base}/oscar_logo_kv_transparent.png"
			alt="OSCAR"
		/>
	</div>

	<div class="min-w-0 leading-none">
		<div class="truncate text-xl font-semibold tracking-[0.08em] text-foreground">OSCAR</div>
		{#if showTagline}
			<div class="mt-1 truncate text-[0.68rem] font-medium uppercase tracking-[0.16em] text-cyan-600">
				2-bit Covariance KV
			</div>
		{/if}
	</div>
</div>
