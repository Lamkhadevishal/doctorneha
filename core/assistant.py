from ragpipeline import RAGPipeline
import time
import sys
import os


# Typing and thinking animations
def thinking_animation(message="👩‍⚕️ Doctor Neha is thinking", delay=0.4, dots=3):
    sys.stdout.write(message)
    sys.stdout.flush()
    for _ in range(dots):
        time.sleep(delay)
        sys.stdout.write(".")
        sys.stdout.flush()
    print()

def type_like_doctor_neha(text, delay=0.02):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()


def chat():
    print("Hey, Neha i am  here to assist you from your future. (Type 'exit' to quit, 'reset' to clear memory)")
    assistant = RAGPipeline(doc_path="C:/Users/comp/Desktop/doctorneha/data/doctorneha.pdf")

    session_id = "default"

    while True:
        question = input("\n🎓 You: ").strip().lower()
        
        # Exit condition
        if question in ["exit", "quit"]:
            print("👋 Goodbye! Have a healthy day ahead.")
            break

        # Reset memory
        elif question == "reset":
            path = f"history/{session_id}.json"
            if os.path.exists(path):
                os.remove(path)
                print("🧠 Memory cleared! Starting fresh...")
            else:
                print("🧹 No memory found to reset.")
            continue

        # Greet on "hi" or "hello"
        elif question in ["hi", "hello", "hii", "hey"]:
            print("👩‍⚕️ Doctor Neha: Hi, myself Doctor Neha — a physiotherapist. How can I help you today?")
            continue

        # Default response from RAG
        thinking_animation()
        response = assistant.ask(question, session_id=session_id)
        print("\n👩‍⚕️ Doctor Neha:\n")
        type_like_doctor_neha(response)
if __name__=="__main__":
    chat()