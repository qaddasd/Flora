import { useEffect, useRef, useState } from "react";
import { LINKS } from "@/lib/datasets";
import { useDemoStore } from "@/lib/demo-store";

const cta =
  "inline-flex w-full items-center justify-center rounded-full border border-white/60 px-6 py-3 text-sm text-white backdrop-blur transition-all duration-200 hover:scale-[1.02] hover:bg-white/10";

export function HeroCtas({ showWalkthrough = true }: { showWalkthrough?: boolean }) {
  const openDemo = useDemoStore((s) => s.openDemo);
  return (
    <div className="mx-auto mt-0 flex w-fit flex-col gap-4">
      {showWalkthrough ? (
        <button type="button" onClick={openDemo} className={cta}>
          Explore the Demo
        </button>
      ) : null}
      <a href={LINKS.paper} target="_blank" rel="noreferrer" className={cta}>
        Read the Paper
      </a>
      <a href={LINKS.code} target="_blank" rel="noreferrer" className={cta}>
        Access the Code
      </a>
      <a href={LINKS.weights} target="_blank" rel="noreferrer" className={cta}>
        Download the Model
      </a>
    </div>
  );
}

export function Hero() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [ready, setReady] = useState(false);
  const [isPortrait, setIsPortrait] = useState(false);

  useEffect(() => {
    const onResize = () => setIsPortrait(window.innerWidth <= window.innerHeight);
    onResize();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  return (
    <section
      id="hero"
      data-page-section="hero"
      className="relative overflow-hidden"
    >
      <div className="absolute inset-x-0 top-16 bottom-0 h-auto w-full object-cover lg:top-0 lg:h-full">
        {!ready ? (
          <img
            src="/images/video-banner.jpg"
            alt=""
            aria-hidden
            draggable={false}
            className="h-full w-full object-cover"
          />
        ) : null}
        <video
          ref={videoRef}
          src={isPortrait ? "/videos/video-banner-mobile.mp4" : "/videos/video-banner.mp4"}
          autoPlay
          muted
          loop
          playsInline
          preload="metadata"
          onLoadedData={() => setReady(true)}
          className={["h-full w-full object-cover", ready ? "opacity-100" : "opacity-0"].join(" ")}
        />
      </div>
      <div className="pointer-events-none absolute inset-0 bg-black/40 xl:bg-transparent" />
      <div className="pointer-events-none absolute top-[96px] right-0 left-0 z-[11] hidden select-none lg:block">
        <span className="absolute left-[23%] -translate-x-1/2 text-xs font-medium tracking-wide text-white/70 sm:text-sm">
          Actual brain activity
        </span>
        <span className="absolute left-[77%] -translate-x-1/2 text-xs font-medium tracking-wide text-white/70 sm:text-sm">
          Predicted brain activity
        </span>
      </div>
      <div className="h-[70vh] min-h-[420px] w-full lg:h-[70vh]" />
      <div className="absolute inset-0 z-10 flex items-start">
        <div className="mx-auto w-full max-w-[1400px] px-6 pt-[calc(84px+80px)] lg:px-10">
          <div className="flex justify-center">
            <div className="w-full max-w-[500px] text-center">
              <h1 className="mt-10 font-display text-[2.25rem] leading-[1.1] font-semibold tracking-tight sm:text-4xl lg:mt-0">
                An AI Model of the Human Brain
              </h1>
              <p className="mt-3 text-base leading-relaxed text-white/70 sm:text-lg">
                Predicting neural responses
                <br />
                to sight, sound and language.
              </p>
              <div className="mt-5 flex justify-center sm:mt-10">
                <HeroCtas />
              </div>
            </div>
          </div>
        </div>
      </div>
      <div className="h-16" />
    </section>
  );
}
