import { createOpenAI } from "@ai-sdk/openai";
import { generateText } from "ai";
import { expect, test } from "vitest";

test("replays one chat completion within its token budget", async () => {
  const tapeUrl = process.env.RHINOTAPE_URL;
  if (!tapeUrl) throw new Error("Start this test with rhinotape run.");

  const control = async (action, body) => {
    const response = await fetch(`${tapeUrl}/__tape__/${action}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const result = await response.json();
    expect(response.ok, JSON.stringify(result)).toBe(true);
    return result;
  };

  await control("load", { cassette: "chat.tape.json", mode: "replay" });
  const openai = createOpenAI({
    baseURL: `${tapeUrl}/v1`,
    apiKey: "rhinotape-replay",
  });
  const { text } = await generateText({
    model: openai.chat("gpt-4o-mini"),
    prompt: "Say hello.",
    maxRetries: 0,
  });
  expect(text).toBe("Hello from the tape.");

  const check = await control("assert", {
    call_count: 1,
    max_tokens: 32,
    no_errors: true,
    no_unused_interactions: true,
  });
  expect(check.ok).toBe(true);
});
