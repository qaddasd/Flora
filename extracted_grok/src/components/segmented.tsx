type Option<T extends string> = { value: T; label: string };

export function Segmented<T extends string>({
  value,
  onChange,
  options,
  className = "",
  disabled = false,
}: {
  value: T;
  onChange: (value: T) => void;
  options: Option<T>[];
  className?: string;
  disabled?: boolean;
}) {
  const idx = Math.max(0, options.findIndex((o) => o.value === value));
  const n = Math.max(1, options.length);
  const inset = 0.25;
  return (
    <div
      className={[
        "relative isolate h-10 overflow-hidden rounded-xl border border-white/15 bg-black/60 p-1 shadow-sm backdrop-blur select-none",
        disabled ? "pointer-events-none opacity-30" : "",
        className,
      ].join(" ")}
    >
      <div
        aria-hidden
        className="absolute top-1 bottom-1 left-1 rounded-lg bg-white/15 shadow-[0_8px_18px_rgba(0,0,0,0.35)] transition-transform duration-200 ease-out motion-reduce:transition-none"
        style={{
          width: `calc((100% - ${2 * inset}rem) / ${n})`,
          transform: `translateX(${idx * 100}%)`,
        }}
      />
      <div className="relative z-10 flex h-full">
        {options.map((opt) => {
          const active = opt.value === value;
          return (
            <button
              key={opt.value}
              type="button"
              aria-pressed={active}
              onClick={() => onChange(opt.value)}
              className={[
                "flex h-full min-w-0 flex-1 items-center justify-center truncate rounded-lg px-3 text-center text-sm font-medium leading-none whitespace-nowrap transition-colors duration-200",
                active ? "text-white" : "text-white/70 hover:text-white",
              ].join(" ")}
            >
              {opt.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
