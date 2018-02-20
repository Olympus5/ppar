
/* Sequential algorithm for converting a text in a digit sequence
*
* PPAR, TP4
*
* A. Mucherino
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc,char *argv[])
{
   int i,n;
   int count,ascii_code;
   char *text;
   char filename[20];
   short notblank,notpoint,notnewline;
   short number,minletter,majletter;
   FILE *input;
   double result = 0.0;
   double decimal = 1.0;
   int tmp = 0;


   // getting started (we suppose that the argv[1] contains the filename related to the text)
   input = fopen(argv[1],"r");

   if (!input) {
      fprintf(stderr,"%s: impossible to open file '%s', stopping\n",argv[0],argv[1]);
      return 1;
   }

   // checking file size
   fseek(input,0,SEEK_END);
   n = ftell(input);
   rewind(input);

   // reading the text
   text = (char*)calloc(n+1,sizeof(char));

   for (i = 0; i < n; i++) {
      text[i] = fgetc(input);
   }

   // converting the text
   count = 0;

   for (i = 0; i < n; i++) {
      ascii_code = (int)text[i];
      notblank =   (ascii_code !=  32);
      notpoint =   (ascii_code !=  46);
      notnewline = (ascii_code !=  10);
      number =     (ascii_code >=  48 && ascii_code <=  57);  // 0-9
      majletter =  (ascii_code >=  65 && ascii_code <=  90);  // A-Z
      minletter =  (ascii_code >=  97 && ascii_code <= 122);  // a-z

      if(notblank && notpoint && notnewline) {
         if(majletter || minletter) {
            tmp++;
         }else {
            if(number) {
               result += tmp * decimal;
            }

            decimal *= 0.1;
         }

      } else {
         while(tmp >= 10) {
            tmp = tmp / 10 + tmp % 10;
         }

         result += tmp * decimal;

         if(tmp > 0) {
            decimal *= 0.1;
            tmp = 0;
         }
      }
   }

   // closing
   free(text);  fclose(input);

   printf("%0.100lf\n", result);

   // ending
   return 0;
}
