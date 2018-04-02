

void SetKernels(Data *data)
{
    int i;
	int s;
	int r;
    
	s = max( round(2 * data->sigma) , 1 );
	r = max( round(2 * data->rho) , 1 ); 
	data->lenSK1 = 2*s +1;
	data->lenSK2 = 2*r +1;

    
    if( data->sigma > 0 )
    {
        data->SKernel1 = (double *)AllocMem(sizeof(double) * data->lenSK1);
        for( i=0 ; i < data->lenSK1 ; i++)
            data->SKernel1[i] = exp( -((i-s)*(i-s))/(2* data->sigma * data->sigma) );
        
        data->Shelp = (double *) AllocMem(sizeof(double) * data->lenSK1);
    }
    
    data->SKernel2 = (double *)AllocMem(sizeof(double) * data->lenSK2);
    for( i=0 ; i < data->lenSK2 ; i++)
        data->SKernel2[i] = exp( -((i-r)*(i-r))/(2* data->rho * data->rho) );
    
}
