Sequential_workflow 
  Non LLM Based
  ----> BMI   input ---> CalBMI---> O/P
  ----> BMI(Enhanched)   input ---> CalBMI ---> label_bmi ---> O/P
  
  LLM Based
  ----> LLM_QA  START--> llm_qa ---> END 
  ----> Prompt Chaning ---> LLM1 Outline --> LLM2 (Topic Generator blog ) --> Final blog
  (START --> Gen_outline --> Gen_blog --END)
  
Parallel_workflow
  Non LLM Based
   --> Cricket Stats
                     
                     START
              .        .       .
            .          .        .
          SR       Boundary %    Boundary Per ball
           .           .          .
             .         .         .
                     Summary
                      END 

  LLM Based
   --> Essay Evaluator 
                     
                                START
                  .               .                    .
                .                 .                      .
    COT(Clarity of thought)     Depth of Analysis %    Language 
              .                  .                        .
                .                .                      .
                              Final Eval
                                  .
                                END 
 

Conditional workflow:
  No LLM
                    START

                show_equation

          calculate_discriminant

no_ral_roots      real_roots      repeated_roots

                   END


  LLM

                    START

                find sentiment

            condn               Condn
      
      run diagnostic            
                                  
    negative_Respone            positive_Response

                   END

Iterative workflow
                    START

                  generate (LLM)

                 Evaluate (LLM2)

condn( Approved )               Condn(needs_imrovement)
      
      END                           optimize (LLM3)

                  
ChatBOt_1

      START

    Chat_Node

      END
