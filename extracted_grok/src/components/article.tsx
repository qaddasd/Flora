import type { ReactNode } from "react";
import {
  FEATURE_CHIPS,
  FIGURES,
  IMPROVEMENT_CARDS,
  RESEARCH_CARDS,
  SECTIONS,
} from "@/lib/content";
import { LINKS } from "@/lib/datasets";
import { useDemoStore } from "@/lib/demo-store";
import { HeroCtas } from "./hero";

function InfoTip({ text }: { text: string }) {
  return (
    <span className="group relative inline-flex">
      <span className="inline-flex h-[18px] w-[18px] cursor-default items-center justify-center rounded-full text-[10px] text-white/75">
        i
      </span>
      <span className="pointer-events-none absolute top-6 left-1/2 z-20 hidden w-[220px] -translate-x-1/2 rounded-xl border border-white/10 bg-black px-3 py-2 text-xs leading-snug text-white/90 shadow-[0_10px_30px_rgba(0,0,0,0.35)] group-hover:block">
        {text}
      </span>
    </span>
  );
}

function renderInline(text: string): ReactNode[] {
  const parts: ReactNode[] = [];
  const re = /\*\*([^*]+)\*\*|\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let i = 0;
  while ((m = re.exec(text))) {
    if (m.index > last) parts.push(text.slice(last, m.index));
    if (m[1]) {
      parts.push(
        <strong key={i++} className="font-semibold text-white">
          {m[1]}
        </strong>,
      );
    } else if (m[2] && m[3]) {
      parts.push(
        <a
          key={i++}
          href={m[3]}
          target="_blank"
          rel="noreferrer"
          className="text-white underline decoration-white/30 underline-offset-4 hover:decoration-white/70"
        >
          {m[2]}
        </a>,
      );
    }
    last = m.index + m[0].length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts;
}

function Body({ text, onChip }: { text: string; onChip: (id: string) => void }) {
  const blocks = text.trim().split(/\n\s*\n/);
  return (
    <div className="mt-8 space-y-6 leading-relaxed text-white/70">
      {blocks.map((block, bi) => {
        const trimmed = block.trim();
        if (trimmed === "[CARDS id=improvements]") {
          return (
            <div key={bi} className="grid gap-4 sm:grid-cols-3">
              {IMPROVEMENT_CARDS.map((c) => (
                <div
                  key={c.id}
                  className="rounded-[18px] border border-white/15 bg-white/5 p-5"
                >
                  <img src={c.icon} alt="" className="mb-3 h-10 w-10 opacity-80" />
                  <h3 className="font-display text-lg font-semibold text-white">{c.label}</h3>
                  <p className="mt-2 text-sm leading-relaxed text-white/65">{c.description}</p>
                </div>
              ))}
            </div>
          );
        }
        if (trimmed === "[CARDS id=research]") {
          return (
            <div key={bi} className="grid gap-4 sm:grid-cols-3">
              {RESEARCH_CARDS.map((c) => (
                <div
                  key={c.id}
                  className="rounded-[18px] border border-white/15 bg-white/5 p-5"
                >
                  <img src={c.icon} alt="" className="mb-3 h-10 w-10 opacity-80" />
                  <h3 className="font-display text-lg font-semibold text-white">{c.label}</h3>
                  <p className="mt-2 text-sm leading-relaxed text-white/65">{c.description}</p>
                </div>
              ))}
            </div>
          );
        }
        if (trimmed === "[CHIPS id=features]") {
          return (
            <div key={bi} className="flex flex-wrap gap-3">
              {FEATURE_CHIPS.map((c) => (
                <button
                  key={c.id}
                  type="button"
                  onClick={() => onChip(c.staticId)}
                  className="group relative flex h-[91px] w-[108px] flex-col items-center justify-center overflow-hidden rounded-[18px] border border-white/45 bg-transparent transition-colors duration-200 hover:border-white"
                >
                  <span className="text-sm font-medium text-white">{c.label}</span>
                  <span className="mt-1">
                    <InfoTip text={c.tooltip} />
                  </span>
                </button>
              ))}
            </div>
          );
        }
        const fig = trimmed.match(/^\[FIGURE id=([^\s\]]+)/);
        if (fig) {
          const f = FIGURES[fig[1]!];
          if (!f) return null;
          return (
            <figure key={bi} className="mx-auto max-w-[720px]">
              <img src={f.src} alt={f.alt} className="w-full" />
            </figure>
          );
        }
        if (/^\d+\.\s/.test(trimmed)) {
          const items = trimmed.split(/\n(?=\d+\.\s)/);
          return (
            <ol key={bi} className="space-y-4">
              {items.map((item, ii) => {
                const m = item.match(/^\d+\.\s([\s\S]+)/);
                return (
                  <li key={ii} className="flex gap-3">
                    <span className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full border border-white/30 text-xs text-white">
                      {ii + 1}
                    </span>
                    <span>{renderInline(m?.[1] ?? item)}</span>
                  </li>
                );
              })}
            </ol>
          );
        }
        return (
          <p key={bi} className="text-[17px] leading-[1.65]">
            {renderInline(trimmed)}
          </p>
        );
      })}
    </div>
  );
}

export function Article() {
  const setMode = useDemoStore((s) => s.setMode);
  const setStaticId = useDemoStore((s) => s.setStaticId);
  const setExpanded = useDemoStore((s) => s.setExpanded);
  const termsAccepted = useDemoStore((s) => s.termsAccepted);

  const goSection = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const onChip = (id: string) => {
    setMode("insilico");
    setStaticId(id);
    if (termsAccepted) setExpanded(true);
    else useDemoStore.setState({ showTerms: true });
  };

  return (
    <div className="space-y-10">
      {SECTIONS.map((section, idx) => (
        <section
          key={section.id}
          id={section.id}
          data-page-section={section.id}
          className="scroll-mt-[84px]"
        >
          <div className="flex items-start justify-between gap-3">
            <h2
              className="cursor-pointer text-left font-display text-2xl font-semibold tracking-tight sm:text-3xl"
              onClick={() => goSection(section.id)}
            >
              {section.title}
            </h2>
            <div className="mt-1.5 flex shrink-0 items-center gap-1">
              <button
                type="button"
                aria-label="Previous section"
                disabled={idx === 0}
                onClick={() => goSection(SECTIONS[idx - 1]!.id)}
                className="section-nav-btn flex h-9 w-9 items-center justify-center rounded-full border border-white/30 text-white/60 transition-colors hover:border-white/60 hover:text-white disabled:cursor-default disabled:opacity-20"
              >
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                  <path
                    d="M8 3.5v9M4 7l4-4 4 4"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </button>
              <button
                type="button"
                aria-label="Next section"
                disabled={idx === SECTIONS.length - 1}
                onClick={() => goSection(SECTIONS[idx + 1]!.id)}
                className="section-nav-btn flex h-9 w-9 items-center justify-center rounded-full border border-white/30 text-white/60 transition-colors hover:border-white/60 hover:text-white disabled:cursor-default disabled:opacity-20"
              >
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                  <path
                    d="M8 12.5v-9M12 9l-4 4-4-4"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </button>
            </div>
          </div>
          <Body text={section.body} onChip={onChip} />
          {section.id === "explore" ? (
            <div className="mt-10">
              <HeroCtas />
            </div>
          ) : null}
        </section>
      ))}
      <p className="pt-8 text-sm text-white/40">
        Paper, code and weights:{" "}
        <a href={LINKS.paper} className="underline underline-offset-4">
          paper
        </a>
        ,{" "}
        <a href={LINKS.code} className="underline underline-offset-4">
          code
        </a>
        ,{" "}
        <a href={LINKS.weights} className="underline underline-offset-4">
          model
        </a>
        .
      </p>
    </div>
  );
}
