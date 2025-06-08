# Mutee: The Volume Whisperer

Welcome to Mutee, the application that listens to your voice and adjusts your audio levels like a personal DJ who’s had one too many cups of coffee! Tired of shouting at your computer to lower the volume? Well, now you can just say a few magic words, and voilà! Your wish is Mutee's command.

## Features

- **Voice-Activated Volume Control**: Just say the trigger phrases, and watch your apps' volumes drop faster than your motivation on a Monday morning.
- **Silence Detection**: Mutee knows when you’re not talking but sometimes mixes it up with mouse clicks.
- **Multi-Platform Support**: Whether you’re on Windows or Linux, Mutee has got your back (sorry, macOS users, you’re on your own).

## Installation Instructions

### For Windows Users

1. **Install Python**: Make sure you have Python installed. If you don’t, go to [python.org](https://www.python.org/downloads/) and download the latest version. Remember to check the box that says "Add Python to PATH" during installation. Otherwise, you’ll be lost in the land of command prompts.

2. **Clone the Repository**: Open your command prompt (you can search for `cmd` in the Start menu) and run:
   ```
   git clone https://github.com/yourusername/mutee.git
   ```
   (Replace `yourusername` with your actual GitHub username, or just download the ZIP file if you prefer the old-school way.)

3. **Navigate to the Project Directory**: Change your directory to the cloned repository:
   ```
   cd mutee
   ```

4. **Create a Virtual Environment (Optional but Recommended)**: This keeps your project dependencies organized. Run:
   ```
   python -m venv venv
   ```
   Then activate it:
   ```
   venv\Scripts\activate
   ```

5. **Install Dependencies**: Now, it’s time to install the required packages. Run:
   ```
   pip install -r requirements.txt
   ```
   If you see any errors, just pretend you didn’t and move on. (Just kidding, fix them!)

6. **Run the Application**: Finally, you can start Mutee by running:
   ```
   python -m __main__
   ```
   Now, start talking to your computer like it’s your best friend!

## Usage

- **Trigger Phrases**: Use phrases like "звука бы", "тих тих тих", or "шумно" to lower the volume of specified applications.
- **Reverse Trigger Phrases**: If you want to revert the volume changes, just say "неважно" or "всё нормально".

## Troubleshooting

- If Mutee doesn’t respond, make sure your microphone is working. You might be talking to yourself again.
- Check the console for any error messages. They might be more helpful than your last therapist.

## Contributing

Feel free to fork the repository and submit pull requests.