#include <ros/ros.h>
#include <std_msgs/Int32.h>
#include <signal.h>
#include <termios.h>
#include <stdio.h>

#define KEYCODE_ENABLE 'a' // a 
#define KEYCODE_NEXT   'b' //b
#define KEYCODE_DISABLE 'q'
#define KEYCODE_QUIT 'p' 

class KeyboardInput
{
public:
  KeyboardInput();
  void keyLoop();

private:
  ros::NodeHandle nh_;
  ros::Publisher key_pub_;
  
};

KeyboardInput::KeyboardInput()
{
  key_pub_ = nh_.advertise<std_msgs::Int32>("rgb_classification/keypressed", 2);
}

int kfd = 0;
struct termios cooked, raw;

void quit(int sig)
{
  (void)sig;
  tcsetattr(kfd, TCSANOW, &cooked);
  ros::shutdown();
  exit(0);
}


void KeyboardInput::keyLoop()
{
  using namespace std;
  char c;
  bool dirty_ = false;
  int keyvalue = 0;


  // get the console in raw mode                                                              
  tcgetattr(kfd, &cooked);
  memcpy(&raw, &cooked, sizeof(struct termios));
  raw.c_lflag &=~ (ICANON | ECHO);
  // Setting a new line, then end of file                         
  raw.c_cc[VEOL] = 1;
  raw.c_cc[VEOF] = 2;
  tcsetattr(kfd, TCSANOW, &raw);

  puts("Reading from keyboard");
  puts("---------------------------");


  for(;;)
  {
    // get the next event from the keyboard  
    if(read(kfd, &c, 1) < 0)
    {
      perror("read():");
      exit(-1);
    }
    cout<<"KeyboardInput Published "<<c<<endl;
  
    switch(c)
    {
      case KEYCODE_ENABLE:
        keyvalue = 1;
        dirty_ = true;
        break;
      case KEYCODE_NEXT:
        keyvalue++;
        dirty_ = true;
        break;
      case KEYCODE_DISABLE:
        keyvalue = 0;
        dirty_ = true;
        break;
      case KEYCODE_QUIT:
        return;
    }
    
    std_msgs::Int32 keypressed_;
    keypressed_.data = keyvalue;
    if (dirty_ ==true)
    {
      key_pub_.publish(keypressed_);  
      dirty_=false;  
    }
    if (keyvalue == 4)
      keyvalue = 0;
  }


  return;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "keyboard_input");
  KeyboardInput keyboard_input;

  signal(SIGINT,quit);

  keyboard_input.keyLoop();
  quit(0);
  
  return(0);
}
