from database import db

class Result(db.Model):

    id = db.Column(db.Integer, primary_key=True)
    student_name = db.Column(db.String(100))
    roll_no = db.Column(db.String(50))
    subject = db.Column(db.String(100))
    obtained_marks = db.Column(db.Integer)
    total_marks = db.Column(db.Integer)
    percentage = db.Column(db.Float)

    def to_dict(self):
        return {
            "id": self.id,
            "studentName": self.student_name,
            "rollNo": self.roll_no,
            "subject": self.subject,
            "obtainedMarks": self.obtained_marks,
            "totalMarks": self.total_marks,
            "percentage": self.percentage
        }