import { useState } from "react";
import { LINKS } from "@/lib/datasets";
import { useDemoStore } from "@/lib/demo-store";

export function CookieModal() {
  const show = useDemoStore((s) => s.showCookies);
  const accept = useDemoStore((s) => s.acceptCookies);
  if (!show) return null;
  return (
    <div className="fixed inset-0 z-[140] flex items-center justify-center overflow-y-auto px-4 py-4">
      <div className="fixed inset-0 bg-black/30 backdrop-blur-[6px]" aria-hidden />
      <div className="relative z-10 my-auto w-full max-w-[620px] shrink-0 rounded-[28px] border border-white/15 bg-black/95 px-6 py-6 text-fg shadow-[0_24px_80px_rgba(0,0,0,0.55)]">
        <div className="flex flex-col gap-4">
          <div>
            <h2 className="text-[22px] leading-tight font-medium">
              Allow the use of cookies from Meta on this browser?
            </h2>
            <p className="mt-2 text-[15px] leading-6 text-white/75">
              To find out more about the use of cookies, see our{" "}
              <a
                href={LINKS.privacy}
                target="_blank"
                rel="noreferrer"
                className="underline decoration-white/35 underline-offset-4 hover:decoration-white/80"
              >
                Privacy Policy
              </a>{" "}
              and{" "}
              <a
                href={LINKS.cookies}
                target="_blank"
                rel="noreferrer"
                className="underline decoration-white/35 underline-offset-4 hover:decoration-white/80"
              >
                Cookies Policy
              </a>
              .
            </p>
          </div>
          <div className="flex flex-col gap-3 sm:flex-row sm:justify-end">
            <button
              type="button"
              onClick={accept}
              className="inline-flex h-12 items-center justify-center rounded-full border border-white/20 bg-transparent px-6 text-[15px] font-medium text-white transition hover:bg-white/10"
            >
              Decline optional cookies
            </button>
            <button
              type="button"
              onClick={accept}
              className="inline-flex h-12 items-center justify-center rounded-full border border-accent px-6 text-[15px] font-medium text-white transition hover:bg-accent/12"
            >
              Allow all cookies
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export function TermsModal() {
  const show = useDemoStore((s) => s.showTerms);
  const accept = useDemoStore((s) => s.acceptTerms);
  const setShow = (v: boolean) => useDemoStore.setState({ showTerms: v });
  if (!show) return null;
  return (
    <div className="fixed inset-0 z-[140] flex items-center justify-center overflow-y-auto px-4 py-4">
      <div
        className="fixed inset-0 bg-black/30 backdrop-blur-[6px]"
        aria-hidden
        onClick={() => setShow(false)}
      />
      <div
        role="dialog"
        aria-labelledby="cookies-modal-title"
        className="relative z-10 my-auto w-full max-w-[620px] shrink-0 rounded-[28px] border border-white/15 bg-black/95 px-6 py-6 text-fg shadow-[0_24px_80px_rgba(0,0,0,0.55)]"
      >
        <h2 id="cookies-modal-title" className="text-[22px] leading-tight font-medium">
          Before we begin the demo
        </h2>
        <div className="mt-5 space-y-4 text-[16px] leading-7 text-white/75">
          <p>This is a research demo and may not be used for any commercial purpose(s).</p>
          <p>
            As with all AI systems, there are inherent risks that the demo may not perform exactly as
            intended.
          </p>
          <p>
            For users accessing this demo in the European Union, we do not hold additional information
            to be able to identify you, but you can contact us using the “How to Contact Meta with
            Questions” section in our privacy policy (available at our{" "}
            <a
              href={LINKS.privacy}
              target="_blank"
              rel="noreferrer"
              className="underline decoration-white/35 underline-offset-4 hover:decoration-white/80"
            >
              Privacy Policy
            </a>
            ) if you have questions about your information rights.
          </p>
          <p>
            I agree to the{" "}
            <a
              href="https://www.facebook.com/legal/terms"
              target="_blank"
              rel="noreferrer"
              className="underline decoration-white/35 underline-offset-4 hover:decoration-white/80"
            >
              Meta Terms of Service
            </a>{" "}
            and agree not to use this demo for any commercial purpose(s).
          </p>
        </div>
        <div className="mt-8 flex justify-end">
          <button
            type="button"
            onClick={() => {
              accept();
              useDemoStore.getState().setExpanded(true);
            }}
            className="inline-flex h-12 items-center justify-center rounded-full border border-accent px-6 text-[15px] font-medium text-white transition hover:bg-accent/12"
          >
            I Agree
          </button>
        </div>
      </div>
    </div>
  );
}

export function MobileBanner() {
  const [hidden, setHidden] = useState(false);
  if (hidden) return null;
  return (
    <div className="fixed inset-x-0 bottom-4 z-40 flex justify-center px-4 lg:hidden">
      <div className="w-full max-w-sm rounded-xl bg-[#1a1a1a] p-4 text-center shadow-lg">
        <p className="text-sm text-white/80">The Brain AI demo is not optimized for small screens.</p>
        <p className="mt-2 text-sm text-white/60">
          For an improved experience, we recommend you try on a larger screen
        </p>
        <div className="mt-4 flex justify-center">
          <button
            type="button"
            onClick={() => setHidden(true)}
            className="rounded-full bg-gradient-to-r from-pink to-purple p-px"
          >
            <span className="block w-32 rounded-full bg-[#1a1a1a] py-2 text-white hover:brightness-110">
              Continue
            </span>
          </button>
        </div>
      </div>
    </div>
  );
}
