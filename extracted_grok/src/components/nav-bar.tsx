import { useState } from "react";
import { Link } from "@tanstack/react-router";
import { LINKS } from "@/lib/datasets";

function ExternalIcon() {
  return (
    <svg
      className="inline-block h-[1.5em] w-[1.5em] text-muted"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <line x1="7" y1="17" x2="17" y2="7" />
      <polyline points="7 7 17 7 17 17" />
    </svg>
  );
}

const navLink =
  "mr-4 inline-flex items-center font-normal text-base underline-offset-4 hover:underline hover:opacity-70 lg:mr-6 xl:mr-10";

export function NavBar({ solid = false }: { solid?: boolean }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <div
        className={[
          "fixed top-0 right-0 left-0 z-40 flex w-full items-center justify-center px-6 py-3 text-fg md:justify-between lg:grid lg:grid-cols-3",
          solid ? "border-b border-white/10 bg-black" : "bg-transparent",
        ].join(" ")}
      >
        <button
          type="button"
          className="absolute left-0 z-50 mx-6 p-1 md:hidden"
          aria-label="Open menu"
          onClick={() => setOpen(true)}
        >
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="3" y1="12" x2="21" y2="12" />
            <line x1="3" y1="6" x2="21" y2="6" />
            <line x1="3" y1="18" x2="21" y2="18" />
          </svg>
        </button>
        <Link to="/" className="leading-tight text-center md:text-start">
          <span className="block text-[18px] leading-[28px] font-bold">TRIBE v2</span>
          <span className="block text-[12px] leading-[150%] font-normal text-muted">
            AI research by Meta
          </span>
        </Link>
        <img
          src="/assets/icn-meta-logo-d08e5867.svg"
          alt="Meta Logo"
          className="mx-auto hidden w-20 lg:block"
        />
        <div className="hidden h-full items-center justify-end md:flex">
          <a href={LINKS.code} target="_blank" rel="noreferrer" className={navLink}>
            Code
            <ExternalIcon />
          </a>
          <a href={LINKS.weights} target="_blank" rel="noreferrer" className={navLink}>
            Weights
            <ExternalIcon />
          </a>
          <a href={LINKS.paper} target="_blank" rel="noreferrer" className={navLink}>
            Paper
            <ExternalIcon />
          </a>
          <a href={LINKS.blog} target="_blank" rel="noreferrer" className={navLink}>
            Blog
            <ExternalIcon />
          </a>
        </div>
      </div>

      {open ? (
        <div className="fixed inset-0 z-[200] bg-black/40 md:hidden" onClick={() => setOpen(false)}>
          <section
            className="absolute top-0 right-0 h-full w-screen bg-surface-3 text-fg shadow-[0px_0px_25px_10px_#00000024]"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-center px-2 py-3" onClick={() => setOpen(false)}>
              <div className="text-center">
                <div className="text-[18px] font-bold">TRIBE v2</div>
                <div className="text-[12px] text-muted">AI research by Meta</div>
              </div>
            </div>
            <nav className="flex flex-col space-y-6 px-8 pt-6 text-xl font-semibold">
              <Link to="/" onClick={() => setOpen(false)}>
                Home
              </Link>
              <a href={LINKS.blog} target="_blank" rel="noreferrer">
                Blog
              </a>
              <a href={LINKS.paper} target="_blank" rel="noreferrer">
                Paper
              </a>
              <Link to="/credits" onClick={() => setOpen(false)}>
                Credits
              </Link>
              <a href={LINKS.privacy} target="_blank" rel="noreferrer">
                Privacy
              </a>
              <a href={LINKS.cookies} target="_blank" rel="noreferrer">
                Cookies
              </a>
            </nav>
          </section>
        </div>
      ) : null}
    </>
  );
}
