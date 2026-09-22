/**
 * camera.ts — 720p Video Stream Capture with requestVideoFrameCallback Frame Dedup.
 */

export interface CameraOptions {
  width?: number
  height?: number
  frameRate?: number
}

export class CameraManager {
  private video: HTMLVideoElement | null = null
  private stream: MediaStream | null = null
  private running = false
  private animId: number | null = null

  async start(
    videoElement: HTMLVideoElement,
    onFrame: (video: HTMLVideoElement, timestampMs: number) => void,
    options: CameraOptions = {}
  ): Promise<void> {
    const width = options.width || 1280
    const height = options.height || 720
    const frameRate = options.frameRate || 30

    this.video = videoElement
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        width: { ideal: width },
        height: { ideal: height },
        frameRate: { ideal: frameRate },
        facingMode: 'user',
      },
    })

    this.video.srcObject = this.stream
    await this.video.play()
    this.running = true

    const loop = (now: DOMHighResTimeStamp) => {
      if (!this.running || !this.video) return
      onFrame(this.video, now)

      // Use requestVideoFrameCallback if supported
      if ('requestVideoFrameCallback' in this.video) {
        ;(this.video as any).requestVideoFrameCallback((_: any, metadata: any) => {
          if (this.running && this.video) {
            onFrame(this.video, metadata.expectedDisplayTime || performance.now())
            this.animId = requestAnimationFrame(loop)
          }
        })
      } else {
        this.animId = requestAnimationFrame(loop)
      }
    }

    this.animId = requestAnimationFrame(loop)
  }

  stop(): void {
    this.running = false
    if (this.animId !== null) {
      cancelAnimationFrame(this.animId)
      this.animId = null
    }
    if (this.stream) {
      for (const track of this.stream.getTracks()) {
        track.stop()
      }
      this.stream = null
    }
    if (this.video) {
      this.video.srcObject = null
      this.video = null
    }
  }
}
