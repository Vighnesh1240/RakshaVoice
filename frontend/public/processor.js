
class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.parts = [];
    this.n = 0;
  }

  process(inputs) {
    const ch = inputs[0]?.[0];

    if (ch) {
      this.parts.push(new Float32Array(ch));
      this.n += ch.length;

      if (this.n >= 4000) {
        const out = new Float32Array(this.n);
        let offset = 0;

        for (const part of this.parts) {
          out.set(part, offset);
          offset += part.length;
        }

        this.port.postMessage(out.buffer, [out.buffer]);
        this.parts = [];
        this.n = 0;
      }
    }

    return true;
  }
}

registerProcessor("pcm-processor", PCMProcessor);
