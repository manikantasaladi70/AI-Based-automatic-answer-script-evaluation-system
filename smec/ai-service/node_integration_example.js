// integration helper: forward files to Python AI service
const axios = require("axios");
const FormData = require("form-data");

async function forwardToAI(files) {
  const fd = new FormData();
  fd.append("answerKey", files.answerKey.data, { filename: files.answerKey.name || "key.png" });
  fd.append("studentScript", files.studentScript.data, { filename: files.studentScript.name || "student.png" });
  const res = await axios.post("http://localhost:5000/evaluate", fd, {
    headers: fd.getHeaders(),
    maxContentLength: Infinity,
    maxBodyLength: Infinity
  });
  return res.data;
}

module.exports = { forwardToAI };
