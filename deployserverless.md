  The RunPod Serverless Endpoint has been deployed using your container image, .env credentials, attached persistent
  volume, and 48GB GPU pool (NVIDIA L40S).
  ──────
  ### 1. Endpoint Configuration Summary
   Field                                                    | Value / Details
  ----------------------------------------------------------|---------------------------------------------------------
   Endpoint Name                                            | LTX-2.5-Video-Gen-L40S
   Endpoint ID                                              | hoxdil79z7nafq
   GPU Pool                                                 | ADA_48_PRO (NVIDIA L40S / RTX 6000 Ada 48GB)
   Container Image                                          | ghcr.io/fx-dove/ltx2.5:latest
   Network Volume                                           | scr3boo7r9 (ltx2.5, 200 GB)
   Region / Datacenter                                      | EU-NL-1 (Direct network volume attach)
   Scale / Workers                                          | 0 min / 2 max (Zero idle cost)
   Idle Timeout                                             | 60 seconds
   FlashBoot                                                | Enabled
   Storage / Output                                         | Cloudflare R2 (cineforge bucket) + RunPod Temp fallback
  ──────
  ### 2. Endpoint API Endpoints

  • Sync Execution:
  POST https://api.runpod.ai/v2/hoxdil79z7nafq/runsync
  • Async Execution:
  POST https://api.runpod.ai/v2/hoxdil79z7nafq/run
  • Status Check:
  GET https://api.runpod.ai/v2/hoxdil79z7nafq/status/{JOB_ID}
  ──────
  ### 3. Example Request (cURL)
    curl -X POST "https://api.runpod.ai/v2/hoxdil79z7nafq/runsync" \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
      -d '{
        "input": {
          "prompt": "Cinematic shot of a robotic dove taking flight in a futuristic neon metropolis, volumetric
  lighting, photorealistic 8k",
          "resolution": "450p",
          "num_frames": 241,
          "fps": 24,
          "num_inference_steps": 40,
          "guidance_scale": 3.0
        }
      }'
  ──────
  ### 4. Example Request (Python)

    import runpod

    runpod.api_key = "YOUR_RUNPOD_API_KEY"
    endpoint = runpod.Endpoint("hoxdil79z7nafq")

    run_request = endpoint.run_sync({
        "prompt": "Golden hour aerial footage over misty pine mountains, cinematic 4k",
        "resolution": "450p",
        "num_frames": 121,
        "fps": 24
    })

    print("Result:", run_request)
                                                     