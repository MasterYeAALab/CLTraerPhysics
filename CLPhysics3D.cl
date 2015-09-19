typedef struct{
    float4 force;
    float4 vel;
    float4 pos;
    float mass;
    float fixed;
    float locked;
    float dummy;
} Particle;



typedef struct{
    float springConstant;
    float damping;
    float restLength;
    float on;
} Spring;



__kernel void tick( __global Particle* particles,float time,__global float4* pos,__global Spring* springs, __global Particle* anchors,__global Particle* attractPoint,float range, float distanceMin, float drag,float density,float gravity,__global float4* linePos,__global float4* lineCol,__global int* lineCount,float lineBetweenDistance) {
    
    
    //printf("Tick\n\n");
    int i = get_global_id(0);
    if(i==0){
        *lineCount = 0;
    }
    __global Particle* a = &particles[i];
    
    __global Particle* b = attractPoint;// mouse
    __global Particle* p = a;
    
    
    //-------------Gravity----------------
    if(gravity != 0.0){
        a->force += (float4)(0,gravity,0,0);
    }
    
    //----------Drag----------------
    p = a;
    p->force -= p->vel*drag;
    
    
    
    
    //----------Spring----------------
    __global Spring *s = springs;
    b = &anchors[i];
    if(s->on != 0.0&&(a->fixed != 1.0|| b->fixed != 1.0) ){
        float4 a2b = a->pos - b->pos;
        float a2bDistance = fast_length(a2b);
        a2b = fast_normalize(a2b);
        float springForce = -( a2bDistance - s->restLength ) * s->springConstant;
        float4 Va2b = a->vel - b->vel;
        float dampingForce = -1*s->damping * ( a2b.x*Va2b.x + a2b.y*Va2b.y);
        float r = springForce + dampingForce;
        a2b *= r;
        if ( a->fixed != 1.0)
            a->force += a2b;
    }
    
    //-----Attraction--------
    b = attractPoint;
    if ( a->fixed != 1.0|| b->fixed != 1.0 ){
        float4 b2a = b->pos - a->pos;
        float distance = fast_length(b2a);
        if ( distance < distanceMin )
            distance = distanceMin;
        
        float force=range/ldexp(distance,density);
        
        b2a = fast_normalize(b2a);
        b2a = b2a*force;
        a->vel += b2a;
    }
    
    //-------ApplyForce--------
    
    if (p->fixed != 1.0f) {
        float4 acc = p->force /(p->mass*time);
        p->vel +=  acc;
        p->pos = p->pos + (p->vel/time);
    }
    pos[i] = p->pos;
    
    
    //------------Calculate lines-------------
    if(i!=0){
        float d = distance(pos[i].xy,pos[i-1].xy);
        //printf("%f\n",d);
        float pVel = fast_length(p->vel);
        if(d<lineBetweenDistance&&pVel>2){
            atomic_inc(lineCount);
            //----deal with linePos and lineCol, lineCount
            linePos[(i-1)*2]= pos[i];
            linePos[(i-1)*2+1] = pos[i-1];
            
//            lineCol[(i-1)*2] = (float4)(0.0f,0.0f,0.0f,0.0001+0.002*pVel);
//            lineCol[(i-1)*2+1] = (float4)(0.0f,0.0f,0.0f,0.0001+0.002*pVel);
            
            lineCol[(i-1)*2].w = 0.0001+0.002*pVel;
            lineCol[(i-1)*2+1].w = 0.0001+0.002*pVel;

            
        }
        else
        {
            linePos[(i-1)*2]= (float4)(0.0f,0.0f,0.0f,0.0f);
            linePos[(i-1)*2+1] = (float4)(0.0f,0.0f,0.0f,0.0f);
            
//            lineCol[(i-1)*2]= (float4)(0.0f,0.0f,0.0f,0.0f);
//            lineCol[(i-1)*2+1] = (float4)(0.0f,0.0f,0.0f,0.0f);
        }
        //printf("%f\t%f\n%f\t%f\n",linePos[(i-1)*2].x,linePos[(i-1)*2].y,linePos[(i-1)*2+1].x,linePos[(i-1)*2+1].y);
        
    }
    
    
    
    //---ClearForce------
    p->force = (float4)(0,0,0,0);
}
