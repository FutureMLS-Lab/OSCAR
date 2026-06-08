<script lang="ts">
	import { OscarBrand, ServerStatus } from '$lib/components/app';
	import { fade } from 'svelte/transition';

	interface Props {
		class?: string;
		message?: string;
	}

	let { class: className = '', message = 'Initializing connection to server...' }: Props = $props();
</script>

<div class="flex h-full items-center justify-center {className}">
	<div class="max-w-md px-6 text-center">
		<div
			class="relative mb-6 overflow-hidden rounded-3xl border border-cyan-100 bg-background/80 px-8 py-7 shadow-sm"
			in:fade={{ duration: 300 }}
		>
			<div class="absolute -right-8 -top-8 h-28 w-28 rounded-full bg-cyan-100/60"></div>
			<div class="absolute -bottom-10 left-8 h-24 w-24 rounded-full bg-sky-100/50"></div>

			<div class="relative">
				<OscarBrand class="justify-center" logoClass="h-16 w-16" />

				<h2 class="mt-5 text-2xl font-semibold tracking-tight text-foreground">
					OSCAR 2-bit Covariance
				</h2>

				<p class="mt-2 text-sm leading-6 text-muted-foreground">
					{message}
				</p>
			</div>
		</div>

		<div class="mt-4">
			<ServerStatus class="justify-center" />
		</div>
	</div>
</div>
