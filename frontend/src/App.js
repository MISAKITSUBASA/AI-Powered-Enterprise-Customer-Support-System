import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim()) {
      alert("Question cannot be empty!");
      return;
    }
    try {
      const response = await axios.post("http://localhost:8000/ask", {
        question: question,
      });
      console.log(response);
      setAnswer(response.data.answer.content);
    } catch (error) {
      console.error("Error:", error);
      alert("An error occurred while fetching the answer.");
    }
  };

  return (
    <div style={{ margin: "2rem" }}>
      <h1>AI Customer Support (macOS)</h1>
      <form onSubmit={handleSubmit}>
        <label>
          Ask something:
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            style={{ marginLeft: "1rem", width: "300px" }}
          />
        </label>
        <button type="submit" style={{ marginLeft: "1rem" }}>
          Ask
        </button>
      </form>
      {answer && (
        <div style={{ marginTop: "1rem" }}>
          <strong>AI Answer:</strong> {answer}
        </div>
      )}
    </div>
  );
}

export default App;
