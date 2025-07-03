# ATP Core Talent 2025
# Core Talent AI Coder Challenge: Camera Movement Detection

**Detecting Significant Camera Movement Using Image Recognition**

---

## Scenario

Imagine you are tasked with building a component for a smart camera system. Your goal is to detect **significant movement**—for example, if someone moves or tilts the camera or if the entire camera is knocked or shifted. This is different from simply detecting moving objects in the scene.

---

## Requirements

1. **Input:**

   * A sequence of images or frames (at least 10-20), simulating a fixed camera, with some frames representing significant camera movement (tilt, pan, large translation), and others showing a static scene or minor background/object motion.
   * You may use public datasets, generate synthetic data, or simulate with your own webcam.

     * Example: [CameraBench Dataset on Hugging Face](https://huggingface.co/datasets/syCen/CameraBench)
2. **Task:**

   * Build an algorithm (**Python preferred**) that analyzes consecutive frames and detects when significant camera movement occurs.
   * Output a list of frames (by index/number) where significant movement is detected.
3. **Expected Features:**

   * **Basic:** Frame differencing or feature matching to detect large global shifts (e.g., using OpenCV’s ORB/SIFT/SURF, optical flow, or homography).
   * **Bonus:** Distinguish between camera movement and object movement within the scene (e.g., use keypoint matching, estimate transformation matrices, etc.).
4. **Deployment:**

   * Wrap your solution in a small web app (**Streamlit, Gradio, or Flask**) that allows the user to upload a sequence of images (or a video), runs the detection, and displays the result.
   * Deploy the app on a public platform (**Vercel, Streamlit Cloud, Hugging Face Spaces**, etc.)
5. **Deliverables:**

   * Public app URL
   * GitHub repo (with code and requirements.txt)
   * README (explaining your approach, dataset, and how to use the app)

     * **Sample README Outline:**

       * Overview of your approach and movement detection logic
       * Any challenges or assumptions
       * How to run the app locally
       * Link to the live app
       * Example input/output screenshots
   * AI Prompts or Chat History (if used for support)

---

## Evaluation Rubric

| Criteria           | Points | Details                                                                                    |
| ------------------ | ------ | ------------------------------------------------------------------------------------------ |
| **Correctness**    | 5      | Accurately detects significant camera movement; low false positives/negatives.             |
| **Implementation** | 5      | Clean code, good use of OpenCV or relevant libraries, modular structure.                   |
| **Deployment**     | 5      | App is online, easy to use, and functions as described.                                    |
| **Innovation**     | 3      | Advanced techniques (feature matching, transformation estimation, clear object vs camera). |
| **Documentation**  | 2      | Clear README, instructions, and concise explanation of method/logic.                       |

---

## Suggested Stack

* **Python** or **C#**
* **OpenCV** for computer vision
* **Streamlit**, **Gradio**, or a **shadcn-powered Vercel site** for quick web UI
* **GitHub** for code repo, **Streamlit Cloud**, **Hugging Face Spaces**, or **Vercel** for deployment

---

# 📋 Candidate Instructions

1. **Fork this repository** (or start your own repository with the same structure).
2. **Implement your movement detection algorithm** in `movement_detector.py`.
3. **Develop a simple web app** (`app.py`) that allows users to upload images/sequences and view detection results.
4. **Deploy your app** on a public platform (e.g., Streamlit Cloud, Hugging Face Spaces, Vercel, Heroku) and **share both your deployed app URL and GitHub repository link**.
5. **Document your work**: Include a `README.md` that explains your approach, how to run your code, and sample results (with screenshots or example outputs).

---

**Deadline:**
🕓 **27.06.2025**

---

**Plagiarism Policy:**

* This must be **individual, AI-powered work**.
* You may use open-source libraries, but you **must cite** all external resources and code snippets.
* Do not submit work copied from others or from the internet without proper acknowledgment.

---

**Good luck! Show us your best hands-on AI skills!**
