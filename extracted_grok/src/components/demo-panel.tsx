import { useEffect, useMemo, useRef, useState } from "react";
import { Menu, Pause, Play, Volume2, VolumeX, X } from "lucide-react";
import {
  INSILICO,
  LEGENDS,
  MODES,
  NATURALISTIC,
  PERFORMANCE,
  RGB,
  STATIC_URLS,
  type ColorMode,
  type DemoMode,
  type SurfaceMode,
} from "@/lib/datasets";
import { MODE_COPY } from "@/lib/content";
import { useDemoStore } from "@/lib/demo-store";
import { BrainViewer } from "./brain-viewer";
import { Segmented } from "./segmented";

function ExpandIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <polyline points="15 3 21 3 21 9" />
      <polyline points="9 21 3 21 3 15" />
      <line x1="21" y1="3" x2="14" y2="10" />
      <line x1="3" y1="21" x2="10" y2="14" />
    </svg>
  );
}

function currentColorsUrl(mode: DemoMode, exampleId: string, staticId: string) {
  if (mode === "naturalistic") {
    return NATURALISTIC.find((e) => e.id === exampleId)?.brainColorsUrl ?? NATURALISTIC[0]!.brainColorsUrl;
  }
  return STATIC_URLS[staticId] ?? STATIC_URLS.places;
}

function VideoDock() {
  const exampleId = useDemoStore((s) => s.exampleId);
  const playing = useDemoStore((s) => s.playing);
  const muted = useDemoStore((s) => s.muted);
  const rate = useDemoStore((s) => s.rate);
  const time = useDemoStore((s) => s.time);
  const duration = useDemoStore((s) => s.duration);
  const setPlaying = useDemoStore((s) => s.setPlaying);
  const setMuted = useDemoStore((s) => s.setMuted);
  const cycleRate = useDemoStore((s) => s.cycleRate);
  const setTime = useDemoStore((s) => s.setTime);
  const setDuration = useDemoStore((s) => s.setDuration);
  const setExampleId = useDemoStore((s) => s.setExampleId);
  const videoRef = useRef<HTMLVideoElement>(null);
  const example = NATURALISTIC.find((e) => e.id === exampleId) ?? NATURALISTIC[0]!;
  const idx = NATURALISTIC.findIndex((e) => e.id === example.id);

  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    v.playbackRate = rate;
    v.defaultPlaybackRate = rate;
  }, [rate, exampleId]);

  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    if (playing) void v.play().catch(() => {});
    else v.pause();
  }, [playing, exampleId]);

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="min-h-0 flex-1 overflow-hidden rounded-[22px] border border-white/10 bg-black">
        <video
          ref={videoRef}
          key={example.id}
          src={example.stimulusSrc}
          poster={example.thumbSrc}
          loop
          autoPlay
          playsInline
          muted={muted}
          preload="auto"
          className="block h-full w-full max-h-full max-w-full object-contain"
          onTimeUpdate={(e) => setTime(e.currentTarget.currentTime)}
          onLoadedMetadata={(e) => setDuration(e.currentTarget.duration || 0)}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
        />
      </div>
      <div className="mt-6 flex items-center gap-4">
        <button
          type="button"
          onClick={() => setPlaying(!playing)}
          className="inline-flex h-12 w-12 items-center justify-center rounded-full border border-white/15 bg-black/70 text-white backdrop-blur transition hover:bg-black/90"
        >
          {playing ? <Pause size={20} /> : <Play size={20} />}
        </button>
        <button
          type="button"
          onClick={() => setMuted(!muted)}
          className="inline-flex h-12 w-12 items-center justify-center rounded-full border border-white/15 bg-black/70 text-white backdrop-blur transition hover:bg-black/90"
          aria-label={muted ? "Unmute" : "Mute"}
        >
          {muted ? <VolumeX size={20} /> : <Volume2 size={20} />}
        </button>
        <button
          type="button"
          onClick={cycleRate}
          className="inline-flex h-12 w-12 items-center justify-center rounded-full border border-white/15 bg-black/70 text-white backdrop-blur transition hover:bg-black/90"
        >
          <span className="text-[15px] leading-none font-semibold">{rate}×</span>
        </button>
        <div className="min-w-0 flex-1">
          <input
            type="range"
            min={0}
            max={Math.max(0.001, duration)}
            step={0.01}
            value={time}
            onChange={(e) => {
              const t = Number(e.target.value);
              const v = videoRef.current;
              if (v) v.currentTime = t;
              setTime(t);
            }}
            className="scrub h-8 w-full cursor-pointer"
          />
        </div>
      </div>
      <div className="mt-4 flex items-center justify-between text-sm text-white/60">
        <button
          type="button"
          disabled={idx <= 0}
          onClick={() => setExampleId(NATURALISTIC[idx - 1]!.id)}
          className="disabled:opacity-30"
        >
          ← Previous video
        </button>
        <span>
          {idx + 1} / {NATURALISTIC.length}
        </span>
        <button
          type="button"
          disabled={idx >= NATURALISTIC.length - 1}
          onClick={() => setExampleId(NATURALISTIC[idx + 1]!.id)}
          className="disabled:opacity-30"
        >
          Next video →
        </button>
      </div>
    </div>
  );
}

function GridPicker({ onPickVideo }: { onPickVideo?: () => void }) {
  const mode = useDemoStore((s) => s.mode);
  const exampleId = useDemoStore((s) => s.exampleId);
  const staticId = useDemoStore((s) => s.staticId);
  const setExampleId = useDemoStore((s) => s.setExampleId);
  const setStaticId = useDemoStore((s) => s.setStaticId);

  const items = useMemo(() => {
    if (mode === "naturalistic") {
      return NATURALISTIC.map((e) => ({
        id: e.id,
        title: e.title,
        thumb: e.thumbSrc,
        kind: "video" as const,
      }));
    }
    const list = mode === "performance" ? PERFORMANCE : mode === "insilico" ? INSILICO : RGB;
    return list.map((e) => ({
      id: e.id,
      title: e.title,
      thumb: "/assets/thumb-ba688aba.png",
      kind: "text" as const,
    }));
  }, [mode]);

  const active = mode === "naturalistic" ? exampleId : staticId;

  return (
    <div className="grid h-full min-h-0 auto-rows-max grid-cols-3 content-start gap-4 overflow-y-auto pr-2">
      {items.map((item) => {
        const on = active === item.id;
        return (
          <button
            key={item.id}
            type="button"
            onClick={() => {
              if (mode === "naturalistic") {
                setExampleId(item.id);
                onPickVideo?.();
              } else {
                setStaticId(item.id);
              }
            }}
            className="group block w-full text-left"
          >
            <div
              className={[
                "relative w-full overflow-hidden rounded-[16px] border bg-black transition-all duration-200",
                on
                  ? "border-white shadow-[inset_0_0_0_3px_rgba(255,255,255,0.55)]"
                  : "border-white/10 hover:border-white/25",
              ].join(" ")}
            >
              <div className="pt-[56.25%]" />
              {item.kind === "text" ? (
                <div className="absolute inset-0 flex items-center justify-center bg-surface px-4 text-center">
                  <span className="text-sm font-medium text-white/85">{item.title}</span>
                </div>
              ) : (
                <>
                  <img
                    src={item.thumb}
                    alt={item.title}
                    draggable={false}
                    className="absolute inset-0 h-full w-full object-cover transition duration-200 group-hover:scale-[1.02]"
                  />
                  <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
                    <div className="flex h-12 w-12 items-center justify-center rounded-full bg-black/55 backdrop-blur">
                      <Play size={20} className="ml-0.5 text-white" />
                    </div>
                  </div>
                </>
              )}
            </div>
          </button>
        );
      })}
    </div>
  );
}

function ModeGuide() {
  const mode = useDemoStore((s) => s.mode);
  const setMode = useDemoStore((s) => s.setMode);
  const copy = MODE_COPY[mode];
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex shrink-0 flex-wrap items-center gap-1.5 pt-1 2xl:gap-4">
        {MODES.map((m) => {
          const on = mode === m.id;
          return (
            <button
              key={m.id}
              type="button"
              onClick={() => setMode(m.id)}
              className={[
                "inline-flex h-[38px] items-center justify-center rounded-full border px-3.5 text-[12.5px] font-medium text-white transition-colors duration-200 2xl:h-[46px] 2xl:px-6 2xl:text-[14px]",
                on
                  ? "border-white bg-white/30"
                  : "border-white/20 bg-transparent hover:border-white/40 hover:bg-white/10",
              ].join(" ")}
            >
              {m.action}
            </button>
          );
        })}
      </div>
      <div className="mt-4 mb-3 max-w-[820px] space-y-3 text-[16px] leading-[1.4] text-white/85 2xl:mt-7 2xl:mb-5 2xl:space-y-5 2xl:text-[17px]">
        {copy?.paragraphs.map((p) => (
          <p key={p.slice(0, 32)}>{p}</p>
        ))}
      </div>
    </div>
  );
}

function BrainStage({ tall }: { tall?: boolean }) {
  const mode = useDemoStore((s) => s.mode);
  const colorMode = useDemoStore((s) => s.colorMode);
  const surface = useDemoStore((s) => s.surface);
  const brainOpen = useDemoStore((s) => s.brainOpen);
  const exampleId = useDemoStore((s) => s.exampleId);
  const staticId = useDemoStore((s) => s.staticId);
  const time = useDemoStore((s) => s.time);
  const duration = useDemoStore((s) => s.duration);
  const loading = useDemoStore((s) => s.loading);
  const setLoading = useDemoStore((s) => s.setLoading);
  const setColorMode = useDemoStore((s) => s.setColorMode);
  const setSurface = useDemoStore((s) => s.setSurface);
  const setBrainOpen = useDemoStore((s) => s.setBrainOpen);
  const openDemo = useDemoStore((s) => s.openDemo);
  const expanded = useDemoStore((s) => s.expanded);
  const setExpanded = useDemoStore((s) => s.setExpanded);
  const menuOpen = useDemoStore((s) => s.menuOpen);
  const setMenuOpen = useDemoStore((s) => s.setMenuOpen);
  const setMode = useDemoStore((s) => s.setMode);

  const url = currentColorsUrl(mode, exampleId, staticId);
  const progress = duration > 0 ? time / duration : 0;
  const showColor = mode === "naturalistic" || mode === "insilico";
  const legend = LEGENDS[mode];

  return (
    <div className={["relative min-h-0 w-full overflow-hidden", tall ? "h-full" : "h-[420px] xl:h-[520px]"].join(" ")}>
      {loading ? (
        <div className="absolute inset-0 z-[20] flex items-center justify-center bg-black/70 backdrop-blur-sm">
          <div className="rounded-full border border-white/15 bg-black/75 px-5 py-2.5 text-[14px] font-medium text-white shadow-[0_10px_30px_rgba(0,0,0,0.35)]">
            Brain data loading
            <span className="animate-ellipsis" />
          </div>
        </div>
      ) : null}
      <BrainViewer
        colorsUrl={url}
        colorMode={showColor ? colorMode : "predicted"}
        surface={surface}
        open={brainOpen}
        progress={mode === "naturalistic" ? progress : 0}
        onLoading={setLoading}
      />
      {legend && mode !== "insilico" ? (
        <div className="absolute top-6 right-6 z-30">
          <img
            src={legend.src}
            alt={`${mode} legend`}
            draggable={false}
            className={`${legend.className} w-auto select-none`}
          />
        </div>
      ) : null}

      <div className="absolute top-6 left-6 z-[40]">
        <button
          type="button"
          onClick={() => (expanded ? setExpanded(false) : openDemo())}
          className="inline-flex h-11 items-center gap-2.5 rounded-full border border-white/15 bg-black/75 px-4 text-[14px] text-white backdrop-blur transition hover:bg-black/90 active:scale-[0.98]"
        >
          {expanded ? (
            <>
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path
                  d="M10 4L6 8l4 4"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
              Show Guide
            </>
          ) : (
            <>
              <ExpandIcon />
              Expand Demo
            </>
          )}
        </button>
      </div>

      {!expanded ? (
        <>
          <div className="absolute top-5 right-5 z-[90]">
            <button
              type="button"
              onClick={() => setMenuOpen(!menuOpen)}
              className="inline-flex h-11 w-11 items-center justify-center rounded-full border border-white/15 bg-black/75 text-white backdrop-blur transition hover:bg-black/90"
              aria-label={menuOpen ? "Close menu" : "Open menu"}
            >
              {menuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
            </button>
          </div>
          {menuOpen ? (
            <div className="absolute inset-0 z-[80]" onClick={() => setMenuOpen(false)}>
              <div
                className="absolute top-[4.75rem] right-4 w-[min(260px,calc(100%-1rem))] rounded-[20px] border border-white/10 bg-surface p-3 shadow-[0_20px_60px_rgba(0,0,0,0.45)]"
                onClick={(e) => e.stopPropagation()}
              >
                <div className="mb-1.5 text-[11px] tracking-[0.14em] text-white/45 uppercase">Mode</div>
                <select
                  value={mode}
                  onChange={(e) => {
                    setMode(e.target.value as DemoMode);
                    setMenuOpen(false);
                  }}
                  className="mb-3 h-11 w-full rounded-xl border border-white/15 bg-black px-3.5 text-[14px] text-white"
                >
                  {MODES.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.label}
                    </option>
                  ))}
                </select>
                {showColor ? (
                  <>
                    <div className="mb-1.5 text-[11px] tracking-[0.14em] text-white/45 uppercase">Color</div>
                    <Segmented
                      value={colorMode}
                      onChange={setColorMode}
                      options={[
                        { value: "true", label: "True" },
                        { value: "compare", label: "Compare" },
                        { value: "predicted", label: "Predicted" },
                      ]}
                      className="mb-3 h-9 rounded-lg p-[3px] [&_button]:px-2 [&_button]:text-[13px]"
                    />
                  </>
                ) : null}
                <div className="mb-1.5 text-[11px] tracking-[0.14em] text-white/45 uppercase">Brain</div>
                <Segmented
                  value={brainOpen ? "open" : "close"}
                  onChange={(v) => setBrainOpen(v === "open")}
                  options={[
                    { value: "open", label: "Open" },
                    { value: "close", label: "Close" },
                  ]}
                  disabled={colorMode === "compare"}
                  className="mb-3 h-9 rounded-lg p-[3px] [&_button]:px-2 [&_button]:text-[13px]"
                />
                <div className="mb-1.5 text-[11px] tracking-[0.14em] text-white/45 uppercase">Surface</div>
                <Segmented
                  value={surface}
                  onChange={(v) => setSurface(v as SurfaceMode)}
                  options={[
                    { value: "normal", label: "Normal" },
                    { value: "inflated", label: "Inflated" },
                  ]}
                  className="h-9 rounded-lg p-[3px] [&_button]:px-2 [&_button]:text-[13px]"
                />
              </div>
            </div>
          ) : null}
        </>
      ) : (
        <div className="absolute inset-x-0 bottom-0 z-30 p-3">
          <div className="grid grid-cols-3 gap-3">
            <Segmented
              value={colorMode}
              onChange={setColorMode}
              options={[
                { value: "true" as ColorMode, label: "True" },
                { value: "compare" as ColorMode, label: "Compare" },
                { value: "predicted" as ColorMode, label: "Predicted" },
              ]}
              className={showColor ? "" : "pointer-events-none opacity-0"}
            />
            <Segmented
              value={surface}
              onChange={setSurface}
              options={[
                { value: "normal" as SurfaceMode, label: "Normal" },
                { value: "inflated" as SurfaceMode, label: "Inflated" },
              ]}
            />
            <Segmented
              value={brainOpen ? "open" : "close"}
              onChange={(v) => setBrainOpen(v === "open")}
              options={[
                { value: "open", label: "Open" },
                { value: "close", label: "Close" },
              ]}
              disabled={colorMode === "compare"}
            />
          </div>
        </div>
      )}
    </div>
  );
}

export function DemoPanel() {
  return (
    <div className="relative h-full min-h-0 w-full overflow-hidden rounded-[28px] border border-white/10 bg-black shadow-[0_20px_80px_rgba(0,0,0,0.55)]">
      <BrainStage />
    </div>
  );
}

export function ExpandedDemo() {
  const expanded = useDemoStore((s) => s.expanded);
  const setExpanded = useDemoStore((s) => s.setExpanded);
  const mode = useDemoStore((s) => s.mode);
  const [showVideo, setShowVideo] = useState(true);

  useEffect(() => {
    setShowVideo(mode === "naturalistic");
  }, [mode]);

  if (!expanded) return null;

  return (
    <div className="fixed inset-0 z-[140] flex flex-col bg-black text-white">
      <div className="flex shrink-0 items-center justify-between px-4 pt-3 pb-2">
        <button
          type="button"
          onClick={() => setExpanded(false)}
          className="inline-flex h-9 items-center gap-1.5 rounded-full border border-white/15 bg-black/70 px-3 text-[13px] text-white backdrop-blur"
        >
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path
              d="M10 4L6 8l4 4"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          Back
        </button>
      </div>
      <div className="grid min-h-0 flex-1 grid-cols-1 overflow-hidden px-3 pb-3 lg:grid-cols-[0.95fr_1.05fr]">
        <div className="relative min-h-[280px] overflow-hidden rounded-[28px] border-white/10 bg-surface-2 lg:min-h-0 lg:border-r">
          <BrainStage tall />
        </div>
        <div className="flex min-h-0 flex-col overflow-hidden bg-black">
          <div className="px-4 pt-4 pb-3 xl:px-8 xl:pt-8 xl:pb-5">
            <ModeGuide />
          </div>
          <div className="min-h-0 flex-1 overflow-hidden px-4 pb-6 xl:px-8 xl:pb-8">
            {mode === "naturalistic" && showVideo ? (
              <div className="flex h-full min-h-0 flex-col">
                <VideoDock />
                <button
                  type="button"
                  onClick={() => setShowVideo(false)}
                  className="mt-6 inline-flex h-12 items-center gap-2 self-start rounded-full border border-white/15 bg-black/70 px-5 text-white backdrop-blur transition hover:bg-black/90"
                >
                  <span className="text-lg leading-none">←</span>
                  <span className="text-[18px]">Back</span>
                </button>
              </div>
            ) : (
              <GridPicker onPickVideo={() => setShowVideo(true)} />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
