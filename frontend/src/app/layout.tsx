import type { Metadata } from "next";
import { IBM_Plex_Sans, IBM_Plex_Sans_Condensed, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";

// One superfamily in three roles. A lab report is institutional infrastructure,
// so the type should read that way: condensed for form headers, sans for prose,
// mono for every number, because lab values are monospace everywhere they are
// actually printed.
const sans = IBM_Plex_Sans({
  variable: "--font-sans",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
});

const condensed = IBM_Plex_Sans_Condensed({
  variable: "--font-condensed",
  subsets: ["latin"],
  weight: ["600", "700"],
});

const mono = IBM_Plex_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
});

export const metadata: Metadata = {
  title: "Oncovision, lab report analysis",
  description:
    "Reads a standard lab report and your own history, and reports what the numbers suggest along with how much the result is worth.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className={`${sans.variable} ${condensed.variable} ${mono.variable} antialiased`}>
        {children}
      </body>
    </html>
  );
}
