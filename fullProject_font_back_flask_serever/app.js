const express = require('express');
const path = require('path');
const app = express();
const bodyParser = require('body-parser');
const multer = require('multer');
const request = require('request');

const fileStorage=multer.diskStorage({
    destination:(req,file,cb)=>{
        cb(null,'images');
    },
    filename:(req,file,cb)=>{
        cb(null,new Date().toJSON().slice(0,13)+'-'+file.originalname);
    }
})
app.use(multer({storage:fileStorage}).single('image'));
app.use(bodyParser.urlencoded({ extended:false}));
app.use(express.static('public'));
app.get('/',(req,res)=>{
    res.sendFile(path.join(__dirname,'views','home.html'));
    
})
app.post('/home',(req,res)=>{
    sex=req.body.sex;
    age=parseFloat(req.body.age);
    chest = parseFloat(req.body.chest_pain_type);
    resting_bp = parseFloat(req.body.resting_bp_s);
    cholesterol = parseFloat(req.body.cholesterol);
    fasting = parseFloat(req.body.fasting_blood_sugar);
    resting_ecg = parseFloat(req.body.resting_ecg);
    max_heart_rate = parseFloat(req.body.max_heart_rate);
    angia = parseFloat(req.body.exercise_angina);
    oldpeak = parseFloat(req.body.oldpeak);
    ST_slope = parseFloat(req.body.ST_slope);
    if (sex=='Male'){
        sex=1
    }
    else if (sex=='Female'){
        sex=0
    }        
    const data = {
        "sex":sex,
        "age":age,
        "chest":chest,
        "resting_bp":resting_bp,
        "cholesterol":cholesterol,
        "fasting":fasting,
        "resting_ecg":resting_ecg,
        "max_heart_rate":max_heart_rate,
        "angia":angia,
        "oldpeak":oldpeak,
        "ST_slope":ST_slope

    }
    console.log(data)
    v=JSON.stringify(data)
    request.post({url:'http://127.0.0.1:5000/flask',json:v},function(error, response,body){
        if (error) {
            console.error('Error:', error);
            res.status(500).send('Error sending POST request');
        } else {
            console.log(body);
            res.send(body)
        }
    })
})
app.get('/home', function(req, res) {
    request('http://127.0.0.1:5000/flask', function (error, response, body) {
        
        console.error('error:', error); // Print the error
        console.log('statusCode:', response && response.statusCode); // Print the response status code if a response was received
        console.log('body:', body); // Print the data received
        res.send(body); //Display the response on the website
      });      
});

app.get('/chatbot', (req, res)=> {
    request('http://127.0.0.1:5000/', function (error, response, body) {
        console.error('error:', error); // Print the error
        console.log('statusCode:', response && response.statusCode); // Print the response status code if a response was received
        
         //Display the response on the website
         res.send(body); //Display the response
      })

})
app.post('/send-message', (req, res) => {
    const message = req.body.message;
    chat={
        "msg": message
    }
    json_chat=JSON.stringify(chat)
    console.log(message); // Print the message
    request.post({
        url: 'http://127.0.0.1:5000/get', // Replace with your Flask server endpoint
        json: json_chat
    }, (error, response, body) => {
        if (error) {
            console.error('Error:', error);
            res.status(500).send('Error sending message to Flask server');
        } else {
            console.log('Response from Flask server:', body);
            res.send(body); // Send Flask server response back to client
        }
    });
})
app.get('/symp', (req, res) => {
    res.sendFile(path.join(__dirname, 'views','symptom.html'))
})
app.post('/symp', (req, res) => {
    symp = [req.body.symp]
    const data = {
        "symp":symp
    }
    console.log(data)
    v=JSON.stringify(data)
    request.post({url:'http://127.0.0.1:5000/symp',json:v},function(error, response,body){
        if (error) {
            console.error('Error:', error);
            res.status(500).send('Error sending POST request');
        } else {
            console.log(body);
            res.send(body)
        }

    })
})

app.get('/img',(req,res)=>{
    res.sendFile(path.join(__dirname, 'views','detection.html'))
})

app.post('/all',(req,res)=>{
    img=req.file
    console.log(img)
    request.post({url:'http://127.0.0.1:5000/corona'},function(error, response,body){
        if (error) {
            console.error('Error:', error);
            res.status(500).send('Error sending POST request');
        } else {
            console.log(body);
            res.send(body)
        }

    })
})
app.listen(3000,(req,res)=>{
    console.log('listening');
})