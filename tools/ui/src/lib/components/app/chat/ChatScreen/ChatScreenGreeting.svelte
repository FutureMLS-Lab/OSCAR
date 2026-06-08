<script lang="ts">
	import { fadeInView } from '$lib/actions/fade-in-view.svelte';
	import { OscarBrand } from '$lib/components/app';
	import { serverStore } from '$lib/stores/server.svelte';
	import KvMemoryCurve from './KvMemoryCurve.svelte';

	interface Props {
		isEmpty: boolean;
	}

	let { isEmpty = false }: Props = $props();
</script>

<div
	class={[
		'pointer-events-none mb-4 hidden px-4 text-center',
		isEmpty && 'pointer-events-auto block!'
	]}
	use:fadeInView={{ duration: 300 }}
>
	<div class="relative mx-auto mb-6 max-w-xl overflow-hidden rounded-3xl border border-cyan-100 bg-background/80 px-8 py-7 shadow-sm">
		<div class="absolute -right-10 -top-10 h-32 w-32 rounded-full bg-cyan-100/60"></div>
		<div class="absolute -bottom-12 left-8 h-28 w-28 rounded-full bg-sky-100/50"></div>

		<div class="relative">
			<OscarBrand class="justify-center" logoClass="h-16 w-16" />

			<h1 class="mt-5 text-2xl font-semibold tracking-tight text-foreground md:text-3xl">
				OSCAR 2-bit Covariance
			</h1>

			<p class="mx-auto mt-2 max-w-md text-sm leading-6 text-muted-foreground md:text-base">
				Offline spectral covariance-aware rotation for compact INT2 KV cache inference.
			</p>

			<KvMemoryCurve />
		</div>
	</div>

	<p class="text-muted-foreground md:text-lg">
		{serverStore.props?.modalities?.audio ? 'Record audio, type a message ' : 'Type a message'} or upload
		files to get started
	</p>
</div>
