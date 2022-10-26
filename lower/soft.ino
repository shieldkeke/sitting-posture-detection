#include <SoftwareSerial.h>
#include <math.h>
SoftwareSerial BT(2, 8);
#define E1 13
#define E2 7
#define S1 A1
#define S2 A0
#define THRESHOLD 500


char mat[20][20];
void setup(){

  pinMode(3, OUTPUT);
  pinMode(4, OUTPUT);
  pinMode(5, OUTPUT);
  pinMode(6, OUTPUT);

  pinMode(9, OUTPUT);
  pinMode(10, OUTPUT);
  pinMode(11, OUTPUT);
  pinMode(12, OUTPUT);
    
  pinMode(A4, OUTPUT);
  pinMode(A5, OUTPUT);

  pinMode(E1, OUTPUT);
  pinMode(E2, OUTPUT);
  
  Serial.begin(9600);
  BT.begin(9600);
}
void output(bool num, int out_idx){
  if (num == 0){
    pinMode(S1, OUTPUT);
    digitalWrite(E1, HIGH);
    digitalWrite(E1, LOW);
    bool v;
    int n = out_idx;
    for (int j=12; j>=9; j--){
      v = n % 2;
      n = n / 2;
      digitalWrite(j, v);    
    }
    digitalWrite(S1, HIGH);
  }
  else{
    pinMode(S2, OUTPUT);
    digitalWrite(E2, HIGH);
    digitalWrite(E2, LOW);
    bool v;
    int n = out_idx;
    for (int j=6; j>=3; j--){
      v = n % 2;
      n = n / 2;
      digitalWrite(j, v);    
    }
    digitalWrite(S2, HIGH);
  }
  
}
void input(bool num, int in_idx, int out_idx){
  int vol_;
  char vol;
  int low, high;
  
  if (num == 0){
    pinMode(S1, INPUT);
    digitalWrite(E1, LOW);
    if (in_idx == -1){
      low = 0;
      high = 9;
    }else{
      low = high = in_idx;
    }
    for (int i= low;i <= high;i++){
      bool v;
      int n = i;
      //read
      for (int j=12; j>=9; j--){
        v = n % 2;
        n = n / 2;
        digitalWrite(j, v);    
      }

      vol_ = analogRead(S1)>=1000 ? 999 : analogRead(S1);

      vol = 0;
      if (vol_ > 200) vol = 1;
      if (vol_ > 500) vol = 2;
      if (vol_ > 700) vol = 3;
      if (vol_ > 800) vol = 4;
      if (vol_ > 825) vol = 5;
      if (vol_ > 850) vol = 6;
      if (vol_ > 875) vol = 7;
      if (vol_ > 900) vol = 8;
      if (vol_ > 925) vol = 9; 
      
      
      //write to mat
      if (out_idx>=0){
        mat[i][out_idx] = vol;
      }
      else{    
        for (int j=0; j<16; j++){
          mat[i][j] = vol;
        }
      } 
    }
  }
  else{
    pinMode(S2, INPUT);
    digitalWrite(E2, LOW);
    if (in_idx == -1){
      low = 0;
      high = 9;
    }else{
      low = high = in_idx;
    }
    for (int i= low;i <= high;i++){
      bool v;
      int n = i;
      for (int j=6; j>=3; j--){
        v = n % 2;
        n = n / 2;
        digitalWrite(j, v);    
      }
      vol = analogRead(S2) > THRESHOLD;
      
      if (out_idx>=0){
        mat[out_idx][i] = vol;
      }
      else{    
        for (int j=0; j<16; j++){
          mat[j][i] =mat[j][i] && vol;
        }
      }

    }
  }
}
void show(){
  

  for (int i=0; i<10; i++)
    for (int j=0; j<15; j++)
    {
      int c = mat[i][j];
//      Serial.print(c);
      BT.print(c);
    }
//  Serial.println();
  BT.println();
}

void BTwork(){
    if(Serial.available()){    
      char ch = Serial.read();
      Serial.println(ch);     
      BT.print(ch);        
    }
    if(BT.available()){
      char ch1 = BT.read();  
      Serial.println(ch1);
    }
}

void loop()
{
  // put your main code here, to run repeatedly:

  digitalWrite(A5, HIGH);
  digitalWrite(A4, LOW);
  for (int j=0; j<=14; j++){
    output(1, j);//3-6
    input(0, -1, j);  
  }
  

  show();
  BTwork();
//  delay(100);
}
