

void setup() {
  Serial.begin(9600);           //  setup serial
}

double linear_slope(double y)
{
  // https://cdn.sparkfun.com/assets/8/a/9/4/b/Current_to_Voltage_45a.png

  double x2 = 40.0;
  double y2 = 2.9;

  double x1 = 0.0;
  double y1 = 0.0;

  double slope = (y2 - y1)/(x2 - x1);

  // y = mx + b
  return y/slope;
}

void loop() {
  int current_sensor = analogRead(A0);
  
  double voltage_for_current = current_sensor * (5 / 1023.0);
  double corespondent_current = linear_slope(voltage_for_current);
  Serial.println(corespondent_current);
}